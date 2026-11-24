#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
upstream_dir="$repo_dir/.upstream"
upstream_url="https://github.com/apple-aiml-research/ml-depth-pro-2.git"
upstream_revision="aa389cbb0b74341664ed663a0966c859905cd6c0"

if [[ ! -d "$upstream_dir/.git" ]]; then
    git init --quiet "$upstream_dir"
fi
if ! git -C "$upstream_dir" cat-file -e "$upstream_revision^{commit}" 2>/dev/null; then
    git -C "$upstream_dir" fetch --depth 1 "$upstream_url" "$upstream_revision"
fi
git -C "$upstream_dir" checkout --detach "$upstream_revision"

mkdir -p "$repo_dir/outputs"
run_dir="$(mktemp -d "$repo_dir/outputs/run.XXXXXXXX")"
cd "$upstream_dir"
uv run depth-pro-2 predict -m depthpro2-s -i assets/example.jpg -o "$run_dir/" --save-npz
test -s "$run_dir/example-depth.png"
test -s "$run_dir/example-depth.npz"
printf 'Smoke test passed. Results are in %s\n' "$run_dir"
