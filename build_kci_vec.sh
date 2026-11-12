#!/bin/sh
# Build the kci_vec library: static (.a) + shared (.so/.dylib) + the example.
# At-rest encryption is AES-256-CTR + CMAC: AES-NI is picked at run time on x86,
# the ARMv8 AES instructions come with -mcpu=native. On macOS the shared lib is ad-hoc codesigned
# (the loader SIGKILLs unsigned dylibs on Apple Silicon).
# static lib, codesign, -fno-strict-aliasing.
set -e
OPT="-O3 -fno-strict-aliasing -std=gnu11"
case "$(uname -m)" in
  arm64|aarch64) MARCH="-mcpu=native" ;;
  *)             MARCH="-march=native -mprefer-vector-width=256" ;;   # (2026-11-03) 256-bit vectors: GCC 15.2 miscompiled the 512-bit path for znver5
esac
CC="${CC:-cc}"
UNAME=$(uname -s)

$CC $OPT $MARCH -c kci_vec.c -o kci_vec.o
ar rcs libkci_vec.a kci_vec.o
echo "built libkci_vec.a"

if [ "$UNAME" = "Darwin" ]; then
  $CC $OPT $MARCH -dynamiclib kci_vec.c -o libkci_vec.dylib -lpthread -lm
  codesign --force --sign - libkci_vec.dylib
  SHARED=libkci_vec.dylib
else
  $CC $OPT $MARCH -shared -fPIC kci_vec.c -o libkci_vec.so -lpthread -lm
  SHARED=libkci_vec.so
fi
echo "built $SHARED"

$CC $OPT $MARCH -I. example_kci_vec.c libkci_vec.a -lpthread -lm -o example_kci_vec
echo "built example_kci_vec"
