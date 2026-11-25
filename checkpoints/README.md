# Local checkpoints

Upload the two Apple checkpoints below to this folder, then run
`bash run-offline.sh` from the repository root after installing Depth Pro 2
with pip.

- [depthpro2-s-bgcc6f8bg3.pt](https://ml-site.cdn-apple.com/models/depth-pro-2/depthpro2-s-bgcc6f8bg3.pt)
- [depthpro2-fov-ew55zg4kq7.pt](https://ml-site.cdn-apple.com/models/depth-pro-2/depthpro2-fov-ew55zg4kq7.pt)

The script links the files into `.upstream/checkpoints/` and writes the
sample results to `outputs/pip-test/`.
