#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
upstream_dir="$repo_dir/.upstream"
cli="$upstream_dir/.venv/bin/depth-pro-2"
checkpoints=(depthpro2-s-bgcc6f8bg3.pt depthpro2-fov-ew55zg4kq7.pt)

if [[ ! -x "$cli" ]]; then
    printf 'Install Depth Pro 2 with pip in .upstream/.venv first. See README.md.\n' >&2
    exit 1
fi

mkdir -p "$upstream_dir/checkpoints"
for name in "${checkpoints[@]}"; do
    source_file="$repo_dir/checkpoints/$name"
    destination="$upstream_dir/checkpoints/$name"
    if [[ -f "$destination" && -s "$destination" ]]; then
        continue
    fi
    if [[ -e "$destination" || -L "$destination" ]]; then
        printf 'Expected a nonempty checkpoint at %s\n' "$destination" >&2
        exit 1
    fi
    if [[ ! -f "$source_file" || ! -s "$source_file" ]]; then
        printf 'Upload the checkpoint to %s\n' "$source_file" >&2
        exit 1
    fi
    ln -s -- "$source_file" "$destination"
done

cd "$upstream_dir"
exec "$cli" predict -m depthpro2-s -i assets/example.jpg \
    -o "$repo_dir/outputs/pip-test/" --save-npz
