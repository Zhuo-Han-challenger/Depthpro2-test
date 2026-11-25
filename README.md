# Depth Pro 2 test

A starter repository for testing Apple's Depth Pro 2 on a sample image.

## Run the smoke test

Install Git and [uv](https://docs.astral.sh/uv/getting-started/installation/) on Linux or an Apple silicon Mac. From the root of this repository, run:

```sh
./smoke-test.sh
```

The script checks out a pinned revision of [Apple's inference code](https://github.com/apple-aiml-research/ml-depth-pro-2) and runs the S model on Apple's sample image. The first run installs Python dependencies and downloads the model weights. Apple's CLI selects CUDA, MPS, or CPU based on availability.

A successful run saves a color depth image and an NPZ file containing metric depth in `outputs/`. The smoke test checks that both files are present and nonempty.

Apple's [model license](https://github.com/apple-aiml-research/ml-depth-pro-2/blob/aa389cbb0b74341664ed663a0966c859905cd6c0/LICENSE_MODEL) limits weight use to noncommercial scientific research and academic development.

## Export a COLMAP reconstruction

Save Depth Pro 2 predictions with Apple's `--save-npz` option. Create a JSON
manifest listing the RGB image and its matching depth NPZ for each frame.
Paths are relative to the manifest, and absolute paths are also accepted.

```json
{
  "frames": [
    {
      "image": "images/example.jpg",
      "depth": "predictions/example-depth.npz"
    }
  ]
}
```

```sh
uv run --script export-colmap.py frames.json /path/to/new-case
```

The exporter writes `cameras.bin`, `images.bin`, `points3D.bin`, and
`points3D.ply` under `gen_video/sparse/0/`. It also writes the rig and frame
metadata used by current COLMAP versions. It samples every eighth row and
column by default. Use `--stride` to change the density and `--max-depth` to
exclude distant finite predictions.

A one-image manifest can use the NPZ focal length and the optical center of
the image, with the camera as the world origin. For multiple images, supply
`intrinsics` as `[fx, fy, cx, cy]` in COLMAP pixel coordinates and
`world_to_camera` as a 3×4 or 4×4 matrix for every frame. Supply undistorted
RGB images and matching depth maps in the same orientation and resolution.
Camera translations must be in meters in a common world frame. You can also
supply these fields for one image to use calibrated geometry.

The COLMAP image name defaults to the input basename. Set `name` per frame
when the consumer expects a different relative image path. Keep the source
images available at those paths for downstream tools. Each projected point
records its source observation. Multi-view feature matching and camera pose
estimation require a separate stage.

## Export Gaussian splats

```sh
uv run --script export-colmap.py frames.json /path/to/new-case --gaussian-ply
```

This adds `gen_video/sparse/0/3dgs.ply` alongside the COLMAP cameras and
point cloud. Each sampled, valid depth pixel initializes one Gaussian in the
same world frame as the exported cameras. The exporter uses the pixel color,
an initial opacity of 0.9, and a camera-aligned footprint based on depth,
focal length, and `--stride`. It uses the degree-zero spherical-harmonic PLY
layout of DA3's default Gaussian exporter.

The splats are a deterministic initialization from the visible surfaces.
Novel-view quality depends on scene coverage and any subsequent Gaussian
optimization. Rendered videos and novel camera trajectories require a renderer.

## Render a preview from the terminal

For an exported case that contains `3dgs.ply`, run this from the repository
root using the existing pip environment:

```sh
.upstream/.venv/bin/python -m pip install matplotlib plyfile 'pycolmap>=4.2,<5'
.upstream/.venv/bin/python render-preview.py frames.json /path/to/case outputs/preview.png
```

The PNG compares the first frame's source image, dense metric depth and two
Gaussian renders. The preview runs on CPU and uses the camera from the
COLMAP export. Use `--shift-m` to change the second camera's offset, which
defaults to 0.65 m to the right.

## Run the sample offline after installing with pip

From the repository root, install Apple's source into a virtual environment:

```sh
(
  cd .upstream &&
  python3 -m venv .venv &&
  .venv/bin/python -m pip install .
)
```

Upload both files listed in [checkpoints](checkpoints/README.md) to the
visible `checkpoints/` folder. The Apple source must already be checked
out under `.upstream/`. Then run:

```sh
bash run-offline.sh
```
