"""Build hook: compile the kci_vec C engine into a shared library and bundle it in the wheel, so `pip install`
gives a ready-to-use in-process engine (ctypes, no server). Metadata lives in pyproject.toml; this file only
adds the compile step.

Needs a C compiler at build time (gcc/clang; set CC to override)."""
import os, sys, platform, subprocess
from setuptools import setup
from setuptools.command.build_py import build_py

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))   # kci_vec.c / .h live at repo root


def _compile_lib(dst_dir):
    src = os.path.join(REPO, "kci_vec.c")
    if not os.path.exists(src):
        raise SystemExit(f"kci_vec.c not found at {src}; build from the repo tree")
    os.makedirs(dst_dir, exist_ok=True)
    cc = os.environ.get("CC") or ("gcc" if sys.platform == "win32" else "cc")
    mach = platform.machine().lower()
    if sys.platform == "win32":
        out = os.path.join(dst_dir, "libkci_vec.dll")
        cmd = [cc, "-O3", "-std=gnu11", "-march=native", "-mprefer-vector-width=256",
               "-shared", "-static", "-static-libgcc", "-I", REPO, "-o", out, src]
    else:
        ext = "dylib" if sys.platform == "darwin" else "so"
        out = os.path.join(dst_dir, "libkci_vec." + ext)
        arch = ["-mcpu=native"] if ("arm" in mach or "aarch64" in mach) else ["-march=native", "-mprefer-vector-width=256"]
        cmd = [cc, "-O3", "-std=gnu11", *arch, "-shared", "-fPIC", "-I", REPO,
               "-o", out, src, "-lpthread", "-lm"]
    print("kci_vec: building shared library:\n  " + " ".join(cmd))
    subprocess.check_call(cmd)
    return out


class BuildWithLib(build_py):
    def run(self):
        super().run()                                   # copy .py into build_lib
        # built in a fixed container and stripped) are already in kci_vec/ and come along as package data; nothing is
        # compiled here, so the wheel does not depend on the building machine's ISA.
        if os.environ.get("KCI_VEC_PREBUILT"):
            print("kci_vec: KCI_VEC_PREBUILT set, bundling the prebuilt libraries")
            return
        _compile_lib(os.path.join(self.build_lib, "kci_vec"))


setup(cmdclass={"build_py": BuildWithLib})
