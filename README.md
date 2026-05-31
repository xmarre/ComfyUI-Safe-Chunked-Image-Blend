# Safe Chunked Image Blend

Safe Chunked Image Blend is a ComfyUI custom node for blending large `IMAGE` tensors with explicit control over resize behavior, compute device, chunk size, and output placement.

It is intended for workflows where the standard image blend path may silently resize one input to match the other, potentially doing a large full-batch CPU resize before blending.

## What it does

The node accepts two ComfyUI `IMAGE` tensors in standard `[B, H, W, C]` format and blends them in chunks.

It can:

- validate image tensor shape, batch size, and channel count
- reject mismatched spatial sizes instead of silently resizing
- resize `image2` to `image1`
- resize `image1` to `image2`
- run resize/blend on CUDA even when inputs arrive as CPU tensors
- return the final result as CPU `float32` for normal ComfyUI compatibility
- process one or more frames per chunk
- preallocate the output tensor instead of concatenating large temporary chunks
- print detailed per-step logs for debugging large tensor workflows

## Why this exists

ComfyUI image tensors are often CPU `float32` tensors by the time they reach post-processing nodes.

The standard Image Blend behavior follows the device of `image1`. If `image1` is on CPU, then the resize and blend also happen on CPU. If the two inputs have different spatial sizes, the node may resize the second input to match the first before blending.

For large upscaling/video workflows, that hidden resize can become expensive or unstable, especially with high-resolution batched tensors.

Safe Chunked Image Blend makes that behavior explicit.

## Nodes

### Safe Chunked Image Blend

Main blend node.

Inputs:

| Input | Description |
|---|---|
| `image1` | First ComfyUI `IMAGE` tensor. Also the target size when using `resize_image2_to_image1`. |
| `image2` | Second ComfyUI `IMAGE` tensor. |
| `blend_factor` | Blend strength. `0.0` keeps `image1`; `1.0` fully uses the selected blend result. |
| `blend_mode` | Blend operation. |
| `resize_policy` | What to do when image sizes differ. |
| `resize_method` | Resize interpolation method. |
| `chunk_size` | Number of frames/images processed per chunk. |
| `compute_device` | Where resize and blend should run. |
| `output_cpu_float32` | Return result on CPU as `float32`. |
| `synchronize_each_chunk` | Synchronize CUDA after each chunk. Useful for debugging and safer execution. |
| `empty_cuda_cache_each_chunk` | Empty CUDA cache after each chunk. Usually leave off unless debugging memory pressure. |
| `log_progress` | Print shape, device, resize, blend, and output logs. |

### Image Pair Shape Probe

Debug helper node.

It accepts two images and prints:

- shape
- dtype
- device

It returns both images unchanged.

Use it to verify what is actually reaching a blend node.

## Blend modes

Supported blend modes:

```text
normal
multiply
screen
add
subtract
difference
darken
lighten
```

## Resize policies

### `error_if_mismatch`

Rejects spatial mismatches.

Use this when you want to catch accidental size mismatches instead of silently resizing.

### `resize_image2_to_image1`

Resizes `image2` to match `image1`.

Use this when `image1` is the target/upscaled branch and `image2` is the lower-resolution/original branch.

### `resize_image1_to_image2`

Resizes `image1` to match `image2`.

Use this when you intentionally want to downscale or match the second input.

## Resize methods

Supported resize methods:

```text
bilinear
bicubic
nearest
area
```

Recommended starting point for large images:

```text
bilinear
```

Use `bicubic` only after the workflow is confirmed stable. Bicubic is heavier.

## Compute device

### `cuda`

Moves each chunk to CUDA for resize and blend.

Recommended for large resize/blend paths when CUDA is available.

### `cpu`

Runs resize and blend on CPU.

This can be useful for small images or when CUDA is unavailable, but it is not recommended for very large upscales.

### `image1`

Uses `image1.device`.

### `image2`

Uses `image2.device`.

## Large CPU resize behavior

When `compute_device=cpu`, the node stays on CPU.
If a resize is needed, CPU chunks are resized with OpenCV (`cv2`).

For very large workloads, prefer `compute_device=cuda` when available.

## Recommended settings for large upscaled image/video workflows

For blending a lower-resolution original/blurred branch into a larger upscaled branch:

```text
resize_policy = resize_image2_to_image1
resize_method = bilinear
chunk_size = 1
compute_device = cuda
output_cpu_float32 = true
synchronize_each_chunk = true
empty_cuda_cache_each_chunk = false
log_progress = true
```

After this works, you can test:

```text
resize_method = bicubic
```

## Installation

Clone or copy this repository into your ComfyUI custom nodes folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YOUR_USERNAME/safe_chunked_image_blend.git
```

Or manually copy the folder:

```text
ComfyUI/custom_nodes/safe_chunked_image_blend
```

Restart ComfyUI.

The node appears as:

```text
Safe Chunked Image Blend
```

The debug helper appears as:

```text
Image Pair Shape Probe
```

## Example

If your upscaled branch is:

```text
image1 = (2, 5464, 3800, 3)
```

and your original/blurred branch is:

```text
image2 = (2, 2732, 1900, 3)
```

then use:

```text
resize_policy = resize_image2_to_image1
compute_device = cuda
chunk_size = 1
```

This processes each frame separately:

```text
image2 chunk -> resize to image1 size -> blend -> copy finished result to output buffer
```

instead of relying on a hidden full-batch resize inside a standard blend node.

## Output format

By default:

```text
output_cpu_float32 = true
```

This returns a normal CPU `float32` ComfyUI `IMAGE` tensor.

This is usually the safest option for compatibility with preview, save, and downstream post-processing nodes.

Set it to `false` only if the next node is known to support CUDA image tensors and you want to keep the result on GPU.

## Debugging

Enable:

```text
log_progress = true
```

Example log output:

```text
[SafeChunkedImageBlend] image1: shape=(2, 5464, 3800, 3) dtype=torch.float32 device=cpu
[SafeChunkedImageBlend] image2: shape=(2, 2732, 1900, 3) dtype=torch.float32 device=cpu
[SafeChunkedImageBlend] settings: blend_mode=normal factor=0.35 resize_policy=resize_image2_to_image1 resize_method=bilinear chunk_size=1 requested_compute_device=cuda effective_compute_device=cuda:0 output_cpu_float32=True
[SafeChunkedImageBlend] chunk 0:1 start
[SafeChunkedImageBlend] chunk 0:1 resizing image2 (1, 2732, 1900, 3) -> (5464, 3800)
[SafeChunkedImageBlend] chunk 0:1 blending
[SafeChunkedImageBlend] chunk 0:1 done
```

If execution stops at a specific line, that line identifies the operation that failed or wedged: copy, resize, blend, synchronize, or output copy.

## Limitations

- Batch sizes must match.
- Channel counts must match.
- The node does not repeat, truncate, or align frames.
- It expects ComfyUI `IMAGE` tensors in `[B, H, W, C]` format.
- It does not fix upstream CUDA, driver, WSL, or model-node instability.
- Very large images can still exceed available memory depending on resolution, chunk size, interpolation mode, and surrounding model memory.

## Requirements

- ComfyUI and PyTorch
- NumPy
- OpenCV (`cv2`) for CPU resize paths

## License

MIT

Add your preferred license here.
