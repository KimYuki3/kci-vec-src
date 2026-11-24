#!/bin/sh
# macOS wheel of kci-vec for the machine's architecture (arm64 or x86_64), libraries built by clang.
# usage (from the repository root):  sh tools/build_wheel_mac.sh <out dir>
set -e
OUT=${1:?out dir}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
ST="$OUT/stage"; rm -rf "$ST"; mkdir -p "$ST/kci_vec"
cp "$ROOT"/clients/python/kci_vec/*.py "$ST/kci_vec/"
cp "$ROOT"/clients/python/pyproject.toml "$ROOT"/clients/python/setup.py "$ROOT"/clients/python/README.md "$ROOT"/LICENSE "$ST/"
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
  export MACOSX_DEPLOYMENT_TARGET=11.0; ISA="-mcpu=apple-m1"; TAG=macosx_11_0_arm64        # every Apple-silicon Mac
else
  export MACOSX_DEPLOYMENT_TARGET=10.13; ISA="-march=haswell -maes"; TAG=macosx_10_13_x86_64   # Intel Macs from 2013 (AVX2)
fi
F="-O3 -std=gnu11 $ISA -shared -fPIC -I$ROOT"
cc $F -o "$ST/kci_vec/libkci_vec.dylib" "$ROOT/kci_vec.c" -lpthread -lm
cc $F -o "$ST/kci_vec/libkci_sparse.dylib" "$ROOT/kci_sparse.c" -lpthread -lm
otool -L "$ST/kci_vec/libkci_vec.dylib" "$ST/kci_vec/libkci_sparse.dylib"
[ -d "$OUT/venv" ] || python3 -m venv "$OUT/venv"
PY="$OUT/venv/bin/python"
"$PY" -m pip -q install --upgrade pip setuptools wheel >/dev/null
rm -rf "$OUT/dist"; mkdir -p "$OUT/dist"
(cd "$ST" && KCI_VEC_PREBUILT=1 "$PY" -m pip -q wheel . --no-deps --no-build-isolation -w "$OUT/dist")
"$PY" -m wheel tags --platform-tag "$TAG" --remove "$OUT"/dist/kci_vec-*-py3-none-any.whl >/dev/null
ls -la "$OUT/dist"; shasum -a 256 "$OUT"/dist/*.whl
