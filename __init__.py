
import gc
import traceback
import numpy as np
import torch
import torch.nn.functional as F

try:
    import cv2
except ImportError:
    cv2 = None

LARGE_CPU_RESIZE_PIXELS = 8_000_000

def _log(message):
    print(f"[SafeChunkedImageBlend] {message}", flush=True)

def _shape(x):
    return tuple(x.shape)

def _validate_image(name, image):
    if not torch.is_tensor(image):
        raise TypeError(f"{name} is not a torch.Tensor: {type(image)!r}")
    if image.ndim != 4:
        raise RuntimeError(f"{name} must be ComfyUI IMAGE tensor [B,H,W,C], got shape={_shape(image)}")
    if image.shape[-1] not in (1, 3, 4):
        raise RuntimeError(f"{name} must have 1, 3, or 4 channels in last dim, got shape={_shape(image)}")
    if image.shape[0] < 1:
        raise RuntimeError(f"{name} has empty batch: shape={_shape(image)}")

def _resolve_requested_device(compute_device, image1, image2):
    if compute_device == "cpu":
        return torch.device("cpu")
    if compute_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("compute_device=cuda was requested, but CUDA is not available")
        return torch.device("cuda:0")
    if compute_device == "image1":
        return image1.device
    if compute_device == "image2":
        return image2.device
    raise RuntimeError(f"Unknown compute_device={compute_device!r}")

def _resize_chunk_nhwc_cpu_cv2(chunk, target_h, target_w, method):
    if cv2 is None:
        raise RuntimeError("CPU resize requires OpenCV/cv2, but cv2 could not be imported")

    if chunk.device.type != "cpu":
        raise RuntimeError(f"CPU OpenCV resize received non-CPU tensor: {chunk.device}")

    if method == "nearest":
        interpolation = cv2.INTER_NEAREST
    elif method == "area":
        interpolation = cv2.INTER_AREA
    elif method == "bilinear":
        interpolation = cv2.INTER_LINEAR
    elif method == "bicubic":
        interpolation = cv2.INTER_CUBIC
    else:
        raise RuntimeError(f"Unsupported resize_method={method!r}")

    source = chunk.detach().contiguous().numpy()
    batch, _h, _w, channels = source.shape
    # ComfyUI IMAGE tensors are handled as float32 NHWC throughout this node.
    resized = np.empty((batch, target_h, target_w, channels), dtype=np.float32)

    for i in range(batch):
        frame = source[i]
        if frame.dtype != np.float32:
            frame = frame.astype(np.float32, copy=False)
        frame = np.ascontiguousarray(frame)
        out = cv2.resize(frame, (target_w, target_h), interpolation=interpolation)
        if channels == 1 and out.ndim == 2:
            out = out[..., None]
        resized[i] = out

    return torch.from_numpy(resized)


def _resize_chunk_nhwc_cuda_torch(chunk, target_h, target_w, method):
    x = chunk.movedim(-1, 1).contiguous()
    if method == "nearest":
        y = F.interpolate(x, size=(target_h, target_w), mode="nearest")
    elif method == "area":
        y = F.interpolate(x, size=(target_h, target_w), mode="area")
    elif method in ("bilinear", "bicubic"):
        y = F.interpolate(x, size=(target_h, target_w), mode=method, align_corners=False)
    else:
        raise RuntimeError(f"Unsupported resize_method={method!r}")
    return y.movedim(1, -1).contiguous()


def _resize_chunk_nhwc(chunk, target_h, target_w, method):
    if chunk.shape[1] == target_h and chunk.shape[2] == target_w:
        return chunk
    if chunk.device.type == "cpu":
        return _resize_chunk_nhwc_cpu_cv2(chunk, target_h, target_w, method)
    return _resize_chunk_nhwc_cuda_torch(chunk, target_h, target_w, method)

def _apply_blend(a, b, factor, blend_mode):
    if blend_mode == "normal":
        return a.mul(1.0 - factor).add_(b, alpha=factor)
    if blend_mode == "multiply":
        mixed = a * b
    elif blend_mode == "screen":
        mixed = 1.0 - (1.0 - a) * (1.0 - b)
    elif blend_mode == "add":
        mixed = a + b
    elif blend_mode == "subtract":
        mixed = a - b
    elif blend_mode == "difference":
        mixed = (a - b).abs()
    elif blend_mode == "darken":
        mixed = torch.minimum(a, b)
    elif blend_mode == "lighten":
        mixed = torch.maximum(a, b)
    else:
        raise RuntimeError(f"Unsupported blend_mode={blend_mode!r}")
    return a.mul(1.0 - factor).add_(mixed, alpha=factor)

class SafeChunkedImageBlend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "blend_factor": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "blend_mode": (["normal", "multiply", "screen", "add", "subtract", "difference", "darken", "lighten"], {"default": "normal"}),
                "resize_policy": (["error_if_mismatch", "resize_image2_to_image1", "resize_image1_to_image2"], {"default": "error_if_mismatch"}),
                "resize_method": (["bilinear", "bicubic", "nearest", "area"], {"default": "bilinear"}),
                "chunk_size": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1}),
                "compute_device": (["cuda", "cpu", "image1", "image2"], {"default": "cuda"}),
                "output_cpu_float32": ("BOOLEAN", {"default": True}),
                "synchronize_each_chunk": ("BOOLEAN", {"default": True}),
                "empty_cuda_cache_each_chunk": ("BOOLEAN", {"default": False}),
                "log_progress": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blend"
    CATEGORY = "image/postprocessing"

    def blend(self, image1, image2, blend_factor, blend_mode, resize_policy, resize_method,
              chunk_size, compute_device, output_cpu_float32, synchronize_each_chunk,
              empty_cuda_cache_each_chunk, log_progress):
        _validate_image("image1", image1)
        _validate_image("image2", image2)

        if image1.shape[0] != image2.shape[0]:
            raise RuntimeError(f"Batch mismatch: image1 batch={image1.shape[0]}, image2 batch={image2.shape[0]}")
        if image1.shape[-1] != image2.shape[-1]:
            raise RuntimeError(f"Channel mismatch: image1 channels={image1.shape[-1]}, image2 channels={image2.shape[-1]}")

        bsz, h1, w1, channels = image1.shape
        _, h2, w2, _ = image2.shape

        resize_1_to = None
        resize_2_to = None
        out_h, out_w = h1, w1

        if (h1, w1) != (h2, w2):
            if resize_policy == "error_if_mismatch":
                raise RuntimeError(
                    "Spatial mismatch. Use resize_image2_to_image1 to preserve image1/upscaled output size, "
                    f"or resize_image1_to_image2 to downscale to image2. image1={_shape(image1)} image2={_shape(image2)}"
                )
            if resize_policy == "resize_image2_to_image1":
                resize_2_to = (h1, w1)
            elif resize_policy == "resize_image1_to_image2":
                resize_1_to = (h2, w2)
                out_h, out_w = h2, w2
            else:
                raise RuntimeError(f"Unknown resize_policy={resize_policy!r}")

        requested_device_name = compute_device
        requested_device = _resolve_requested_device(compute_device, image1, image2)

        resize_needed = resize_1_to is not None or resize_2_to is not None
        pixels_per_output_frame = int(out_h) * int(out_w)
        if resize_needed and requested_device.type == "cpu" and pixels_per_output_frame >= LARGE_CPU_RESIZE_PIXELS and log_progress:
            _log(
                f"keeping compute_device=cpu for large resize ({out_w}x{out_h}, "
                f"{pixels_per_output_frame:,} px/frame); CPU resize uses OpenCV, not torch interpolate"
            )
        device = requested_device

        factor = float(blend_factor)
        chunk_size = int(chunk_size)

        if log_progress:
            _log(f"image1: shape={_shape(image1)} dtype={image1.dtype} device={image1.device}")
            _log(f"image2: shape={_shape(image2)} dtype={image2.dtype} device={image2.device}")
            _log(
                f"settings: blend_mode={blend_mode} factor={blend_factor} resize_policy={resize_policy} "
                f"resize_method={resize_method} chunk_size={chunk_size} requested_compute_device={requested_device_name} "
                f"effective_compute_device={device} output_cpu_float32={output_cpu_float32}"
            )
            _log(f"planned output: shape=({bsz}, {out_h}, {out_w}, {channels}), float32_bytes={bsz*out_h*out_w*channels*4:,}")

        result_device = torch.device("cpu") if output_cpu_float32 else device
        result = torch.empty((bsz, out_h, out_w, channels), dtype=torch.float32, device=result_device)

        old_grad = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        try:
            for start in range(0, bsz, chunk_size):
                end = min(start + chunk_size, bsz)
                if log_progress:
                    _log(f"chunk {start}:{end} start")
                a = b = out = None
                try:
                    a = image1[start:end].to(device=device, dtype=torch.float32, non_blocking=False)
                    b = image2[start:end].to(device=device, dtype=torch.float32, non_blocking=False)
                    if log_progress:
                        _log(f"chunk {start}:{end} copied: a={_shape(a)} {a.device}, b={_shape(b)} {b.device}")
                    if resize_1_to is not None:
                        if log_progress:
                            _log(f"chunk {start}:{end} resizing image1 {_shape(a)} -> {resize_1_to} via {'OpenCV CPU' if a.device.type == 'cpu' else 'torch'}")
                        a = _resize_chunk_nhwc(a, resize_1_to[0], resize_1_to[1], resize_method)
                        if log_progress:
                            _log(f"chunk {start}:{end} resized image1 -> {_shape(a)}")
                    if resize_2_to is not None:
                        if log_progress:
                            _log(f"chunk {start}:{end} resizing image2 {_shape(b)} -> {resize_2_to} via {'OpenCV CPU' if b.device.type == 'cpu' else 'torch'}")
                        b = _resize_chunk_nhwc(b, resize_2_to[0], resize_2_to[1], resize_method)
                        if log_progress:
                            _log(f"chunk {start}:{end} resized image2 -> {_shape(b)}")
                    if a.shape != b.shape:
                        raise RuntimeError(f"Internal shape mismatch after resize: a={_shape(a)} b={_shape(b)}")
                    if log_progress:
                        _log(f"chunk {start}:{end} blending")
                    out = _apply_blend(a, b, factor, blend_mode).clamp_(0.0, 1.0)
                    if synchronize_each_chunk and out.is_cuda:
                        if log_progress:
                            _log(f"chunk {start}:{end} synchronizing CUDA")
                        torch.cuda.synchronize(out.device)
                    if output_cpu_float32:
                        if log_progress:
                            _log(f"chunk {start}:{end} copying result to CPU output buffer")
                        result[start:end].copy_(out.detach().cpu(), non_blocking=False)
                    else:
                        if log_progress:
                            _log(f"chunk {start}:{end} copying result to output buffer")
                        result[start:end].copy_(out.detach(), non_blocking=False)
                    if log_progress:
                        _log(f"chunk {start}:{end} done")
                except BaseException:
                    _log(f"chunk {start}:{end} failed")
                    traceback.print_exc()
                    raise
                finally:
                    del a, b, out
                    gc.collect()
                    if device.type == "cuda":
                        if empty_cuda_cache_each_chunk:
                            torch.cuda.empty_cache()
                        if synchronize_each_chunk:
                            torch.cuda.synchronize(device)
        finally:
            torch.set_grad_enabled(old_grad)

        if log_progress:
            _log(f"returning result: shape={_shape(result)} dtype={result.dtype} device={result.device}")
        return (result,)

class ImagePairShapeProbe:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image1": ("IMAGE",), "image2": ("IMAGE",), "label": ("STRING", {"default": "pair"})}}

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image1", "image2")
    FUNCTION = "probe"
    CATEGORY = "debug"

    def probe(self, image1, image2, label):
        _validate_image("image1", image1)
        _validate_image("image2", image2)
        _log(f"{label}.image1: shape={_shape(image1)} dtype={image1.dtype} device={image1.device}")
        _log(f"{label}.image2: shape={_shape(image2)} dtype={image2.dtype} device={image2.device}")
        return (image1, image2)

NODE_CLASS_MAPPINGS = {
    "SafeChunkedImageBlend": SafeChunkedImageBlend,
    "ImagePairShapeProbe": ImagePairShapeProbe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SafeChunkedImageBlend": "Safe Chunked Image Blend",
    "ImagePairShapeProbe": "Image Pair Shape Probe",
}
