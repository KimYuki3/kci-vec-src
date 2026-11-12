FROM ann-benchmarks

# Build libkci_vec.so from the KCI-Vec sources. Copy kci_vec.c / kci_vec.h next to this Dockerfile
# (ann_benchmarks/algorithms/kci_vec/) before building. (2026-11-09) ann-benchmarks builds the image with the
# repository root as the context, so the COPY paths are relative to it (a bare "kci_vec.c" failed on AWS).
# Flags as in build_kci_vec.sh (-mprefer-vector-width=256: the GCC 15 Zen 5 miscompile).
COPY ann_benchmarks/algorithms/kci_vec/kci_vec.c ann_benchmarks/algorithms/kci_vec/kci_vec.h /tmp/kci/
RUN cd /tmp/kci && \
    gcc -O3 -march=native -mprefer-vector-width=256 -fno-strict-aliasing -std=gnu11 -shared -fPIC \
        kci_vec.c -o /usr/local/lib/libkci_vec.so -lpthread -lm && \
    ldconfig

ENV KCI_VEC_LIB=/usr/local/lib/libkci_vec.so

# ann-benchmarks imports the module; nothing else to install (pure ctypes).
RUN python3 -c "import ctypes; ctypes.CDLL('/usr/local/lib/libkci_vec.so'); print('libkci_vec OK')"
