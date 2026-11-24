"""Run every ```python block of an API document, top to bottom, in one namespace (as a reader would type them)."""
import re, sys, os, tempfile, traceback
src = open(sys.argv[1], encoding="utf-8").read()
skip_text = len(sys.argv) > 2 and sys.argv[2] == "notext"
blocks = re.findall(r"```python\n(.*?)```", src, flags=re.S)
os.chdir(tempfile.mkdtemp(prefix="apidoc_"))
ns = {}; ok = 0
for i, b in enumerate(blocks):
    if skip_text and ("TextIndex" in b or "ix." in b or "ix2" in b):
        print("block %d: skipped (TextIndex)" % i); continue
    try:
        exec(compile(b, "block%d" % i, "exec"), ns); ok += 1
        print("block %d: ok (%d lines)" % (i, b.count("\n")))
    except Exception:
        print("block %d: FAILED\n%s\n%s" % (i, b, traceback.format_exc())); sys.exit(1)
import kci_vec
print("all", ok, "blocks ran | kci_vec from", os.path.dirname(kci_vec.__file__))
for name in ("hits", "found", "w", "r", "d", "new_id"):
    if name in ns:
        x = ns[name]; print(name, "=", (x[:2] if isinstance(x, list) else getattr(x, "shape", x)))
