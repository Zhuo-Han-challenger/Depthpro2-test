"""Render a single Depth Pro 2 RGB-D frame with the released CUDA TMPI model."""

import argparse
import math
import sys
import tempfile
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.nn import functional as F


def prepare_input(image_path, depth_path, max_side, device):
    with Image.open(image_path) as image:
        rgb = np.array(ImageOps.exif_transpose(image).convert("RGB"))
    with np.load(depth_path, allow_pickle=False) as saved:
        depth = saved["depth"]
    height, width = rgb.shape[:2]
    if depth.shape != (height, width):
        raise ValueError("Depth Pro 2 depth must match the oriented RGB image size")
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        raise ValueError("the depth map contains no positive finite samples")
    disparity = np.zeros(depth.shape, dtype=np.float32)
    np.divide(1.0, depth, out=disparity, where=valid)
    np.clip(disparity, 0.01, 100.0, out=disparity)

    rgb = torch.from_numpy(rgb).to(device=device, dtype=torch.float32)
    rgb = rgb.permute(2, 0, 1).unsqueeze(0).contiguous() / 255.0
    disparity = torch.from_numpy(disparity).to(device)[None, None]
    scale = min(1.0, max_side / max(height, width))
    out_height, out_width = int(height * scale), int(width * scale)
    if min(out_height, out_width) < 2:
        raise ValueError("image must be at least 2 pixels on each side after resizing")
    if (out_height, out_width) != (height, width):
        size = out_height, out_width
        rgb = F.interpolate(rgb, size=size, mode="bilinear", align_corners=False)
        disparity = F.interpolate(disparity, size=size, mode="bilinear", align_corners=False)
    lo, hi = disparity.amin(), disparity.amax()
    if hi <= lo:
        raise ValueError("the clipped disparity map has no variation")
    return rgb, (disparity - lo) / (hi - lo)


def prepare_tiles(rgb, disparity, config):
    height, width = rgb.shape[-2:]
    tile = 2 ** math.ceil(math.log2(max(1, config.tilesz2w_ratio * width - 1)))
    tile = min(config.tilesz_max, max(config.tilesz_min, tile))
    step = tile - int(tile * config.padsz2tile_ratio)
    y = torch.arange(0, height, step, device=rgb.device)
    x = torch.arange(0, width, step, device=rgb.device)
    sy, sx = torch.meshgrid(y, x, indexing="ij")
    sx, sy = sx.reshape(1, -1), sy.reshape(1, -1)

    def tiles(tensor):
        padded = F.pad(tensor, (0, tile, 0, tile), mode="replicate")
        windows = padded.unfold(2, tile, step).unfold(3, tile, step)
        windows = windows[:, :, :len(y), :len(x)].permute(0, 2, 3, 1, 4, 5)
        return windows.reshape(1, -1, tensor.shape[1], tile, tile).contiguous()

    # Use the fixed intrinsics from Appendix D.2 of the Depth Pro 2 paper.
    focal = 0.58 * width
    camera = rgb.new_tensor([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
    intrinsics = camera.repeat(1, sx.numel(), 1, 1)
    intrinsics[:, :, 0, 2] -= sx
    intrinsics[:, :, 1, 2] -= sy
    return tiles(rgb), tiles(disparity), intrinsics, sx, sy


def encode_video(renderer, mpis, planes, intrinsics, sx, sy, output, frames):
    output.parent.mkdir(parents=True, exist_ok=True)
    height, width = renderer.ho, renderer.wo
    # H.264 with 4:2:0 chroma needs even image dimensions.
    width, height = width - width % 2, height - height % 2
    with tempfile.TemporaryDirectory(dir=output.parent) as scratch:
        staged = Path(scratch) / "render.mp4"
        with av.open(staged, mode="w", format="mp4") as container:
            stream = container.add_stream("libx264", rate=30)
            stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
            # Use the TMPI demo's translating loop and PyTorch renderer frame order.
            for index in range(frames - 1, -1, -1):
                phase = 2 * math.pi * index / (frames - 1)
                pose = torch.eye(4, device=mpis.device)
                pose[0, 3] = 0.03 * math.sin(phase)
                pose[1, 3] = 0.03 * math.cos(phase)
                pose[2, 3] = 0.03 * (math.cos(phase) - 1)
                rgb = renderer.render(mpis, planes, pose, intrinsics.clone(), sx, sy)
                rgb = rgb[0, :, :height, :width].permute(1, 2, 0)
                rgb = (rgb.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
                frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
                container.mux(stream.encode(frame))
            container.mux(stream.encode())
        staged.replace(output)


def render(args):
    weights_path = args.tmpi_dir.resolve() / "weights" / "mpti_04.pth"
    if not weights_path.is_file():
        raise FileNotFoundError(f"clone the official TMPI repository first: missing {weights_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment")

    sys.path.insert(0, str(weights_path.parent.parent))
    import config
    from tmpi import TMPI
    from tmpi_renderer import TMPIRenderer

    device = torch.device("cuda:0")
    with torch.inference_mode():
        rgb, disparity = prepare_input(args.image, args.depth, args.max_side, device)
        rgb_tiles, disparity_tiles, intrinsics, sx, sy = prepare_tiles(rgb, disparity, config)
        network = TMPI(num_planes=config.num_planes)
        weights = torch.load(weights_path, map_location="cpu", weights_only=True)
        network.load_state_dict({name.removeprefix("module."): tensor for name, tensor in weights.items()})
        network = network.to(device).eval()
        height, width = rgb.shape[-2:]
        print(f"Rendering {args.frames} frames at {width} × {height} with CUDA", flush=True)
        mpis, planes = network(rgb_tiles, disparity_tiles, rgb, disparity)
        renderer = TMPIRenderer(height, width, device=device)
        encode_video(renderer, mpis, planes, intrinsics, sx, sy, args.output, args.frames)
    print(f"Wrote {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="original RGB image")
    parser.add_argument("depth", type=Path, help="Depth Pro 2 metric depth NPZ")
    parser.add_argument("output", type=Path, help="output MP4 path")
    parser.add_argument("--tmpi-dir", type=Path, default=Path(".tmpi"), help="official TMPI checkout")
    parser.add_argument("--max-side", type=int, default=4096, help="maximum length of the long side")
    parser.add_argument("--frames", type=int, default=90, help="number of frames in the camera loop")
    args = parser.parse_args()
    if args.max_side < 2 or args.frames < 2:
        parser.error("--max-side and --frames must both be at least 2")
    try:
        render(args)
    except (OSError, ValueError, KeyError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"TMPI render failed: {exc}\n")


if __name__ == "__main__":
    main()
