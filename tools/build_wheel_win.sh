#!/bin/sh
# Windows wheel (win_amd64) of kci-vec, with the libraries built by MinGW-w64 for AVX2 + FMA + AES.
# usage (Git Bash, from the repository root):  sh tools/build_wheel_win.sh <out dir>
# Needs gcc (MinGW-w64, x86_64) and python on PATH. Nothing is installed outside <out dir>.
set -e
OUT=${1:?out dir}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
ST="$OUT/stage"; rm -rf "$ST"; mkdir -p "$ST/kci_vec"
cp "$ROOT"/clients/python/kci_vec/*.py "$ST/kci_vec/"
cp "$ROOT"/clients/python/pyproject.toml "$ROOT"/clients/python/setup.py "$ROOT"/clients/python/README.md "$ROOT"/LICENSE "$ST/"
# one ISA level: x86-64-v3 (AVX2, FMA, F16C, BMI2) + AES. -static: no MinGW runtime DLLs beside the wheel.
F="-O3 -std=gnu11 -march=x86-64-v3 -maes -mprefer-vector-width=256 -shared -static -static-libgcc -s -I$ROOT"
gcc $F -o "$ST/kci_vec/libkci_vec_avx2.dll" "$ROOT/kci_vec.c"
gcc $F -o "$ST/kci_vec/libkci_sparse.dll" "$ROOT/kci_sparse.c"
# the libraries must not need anything but Windows' own DLLs
if objdump -p "$ST/kci_vec/libkci_vec_avx2.dll" "$ST/kci_vec/libkci_sparse.dll" | grep "DLL Name" | grep -v -i "KERNEL32\|ADVAPI32\|api-ms-win-crt\|msvcrt\|bcrypt"; then
  echo "unexpected DLL dependency" >&2; exit 1
fi
[ -d "$OUT/venv" ] || python -m venv "$OUT/venv"
PY="$OUT/venv/Scripts/python.exe"
"$PY" -m pip -q install --upgrade pip setuptools wheel >/dev/null
rm -rf "$OUT/dist"; mkdir -p "$OUT/dist"
(cd "$ST" && KCI_VEC_PREBUILT=1 "$PY" -m pip -q wheel . --no-deps --no-build-isolation -w "$OUT/dist")
"$PY" -m wheel tags --platform-tag win_amd64 --remove "$OUT"/dist/kci_vec-*-py3-none-any.whl >/dev/null
ls -la "$OUT/dist"
"$PY" - "$OUT/dist" <<'PY'
import sys, glob, zipfile, hashlib
for w in glob.glob(sys.argv[1] + "/*.whl"):
    print(w.replace("\\", "/").split("/")[-1], "sha256", hashlib.sha256(open(w, "rb").read()).hexdigest())
    for n in zipfile.ZipFile(w).namelist(): print("   ", n)
PY
