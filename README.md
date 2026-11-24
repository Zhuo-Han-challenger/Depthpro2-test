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
