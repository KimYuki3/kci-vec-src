# Search your texts by handing them over — the first ten minutes

Put in your own texts (rules, manuals, minutes, notes), ask in plain words, and get the texts that fit.
No knowledge of vectors or models is needed: type the steps below in order. The texts do not leave your machine.

These steps were typed top to bottom in a fresh environment (2026-11-22, kci-vec 0.2.12, WSL2 with Ubuntu 24.04,
Python 3.12). `kci_vec.text` is experimental: names and arguments may change in later versions.

## What you need

| | |
|---|---|
| Machine | Linux (x86-64, Ubuntu 22.04 or later). On Windows: inside WSL2. macOS is not supported yet |
| Python | 3.10 or later (tried with 3.12) |
| Disk | about 2 GB (the tools 1.5 GB, the language models 0.15 to 0.7 GB) |
| Network | once, for the tools and the language models. After that it works offline (to be sure, run with `export HF_HUB_OFFLINE=1`: nothing is looked up on the network then) |

## 1. Install (one to a few minutes)

```sh
python3 -m venv venv
source venv/bin/activate
pip install https://github.com/KimYuki3/kci-vec-dist/releases/download/v0.2.12/kci_vec-0.2.12-py3-none-manylinux_2_35_x86_64.whl
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers
```

- `source venv/bin/activate` is typed again in every new terminal.
- Replace the version (`0.2.12`) by the newest on the [releases page](https://github.com/KimYuki3/kci-vec-dist/releases).
- `torch` is large (about 200 MB for the CPU build). The line above does not fetch the GPU parts, which keeps it small.

## 2. The first search

Save as `first.py`:

```python
from kci_vec.text import TextIndex

ix = TextIndex("en", sparse="bm25")
ix.add([
    "Paid leave: 10 days are granted after six months of service. Apply three business days ahead.",
    "Expense reports close at the end of the month. Hand the original receipts to general affairs.",
    "If you forget your password, call the IT desk at extension 1234.",
    "Meeting rooms are booked in the company calendar, for up to two hours at a time.",
])
for hit in ix.search("how many days of paid leave do I get?", k=2):
    print(hit.id, round(hit.score, 3), hit.text)
```

```sh
python first.py
```

It prints (the first run also downloads a language model of a few hundred MB; later runs take seconds):

```
0 1.0 Paid leave: 10 days are granted after six months of service. Apply three business days ahead.
1 0.094 Expense reports close at the end of the month. Hand the original receipts to general affairs.
```

- The number on the left (`hit.id`) is the position of the text as you put it in, from 0.
- The middle one (`hit.score`) only orders the answers: larger fits better. Look at the gap to the first; 0.094 is
  "the best of the rest", not a match.
- Lines like `Loading weights ...` are not errors.

## 3. Your own files, saved

Put your text files (`.txt`) into a folder `docs`. `build.py`:

```python
import glob
from kci_vec.text import TextIndex

paths = sorted(glob.glob("docs/*.txt"))
texts = [open(p, encoding="utf-8").read() for p in paths]

ix = TextIndex("en", sparse="bm25")
ix.add(texts)                                   # numbered 0, 1, 2, ... in this order
ix.save("my_index", key="your passphrase here")
print(len(ix), "texts saved")
```

- Long texts are split by themselves. An answer is a text; `hit.text` is the part of it that fits best.
- With `key=`, the index and the texts are saved encrypted: the saved folder shows no text.
- A forgotten passphrase cannot be recovered. It is not stored anywhere.

## 4. Open it again and ask

`ask.py`:

```python
import glob, sys
from kci_vec.text import TextIndex

paths = sorted(glob.glob("docs/*.txt"))
ix = TextIndex.load("my_index", key="your passphrase here")
for hit in ix.search(sys.argv[1], k=3):
    print(paths[hit.id], round(hit.score, 3))
    print("   ", hit.text.replace("\n", " ")[:80])
```

```sh
python ask.py "what is the limit for a hotel night?"
```

- A wrong passphrase is refused (`RuntimeError`); nothing is opened.
- After adding or removing files in `docs`, run `build.py` again (the order of the files gives the numbers).

## Things you will use

```python
ix.add("Working from home is allowed two days a week.")   # one more text later (numbered on from the last)
ix.add(texts, ids=[101, 102, 103])                        # with numbers of your own
ix.delete(3)                                              # remove number 3
ix.search("leave", k=5, filter=lambda i: i != 0)          # restrict by number (a function, or a set of numbers)
hit.span                                                  # where hit.text sits in the whole text (start, end)
ix.texts[hit.id]                                          # the whole text
```

## Where people stumble

| What happens | Why, and what to do |
|---|---|
| `pip install` says no matching distribution | The wheel is for Linux (x86-64) only. On Windows, do the same steps inside WSL2 |
| The very first run is slow | It downloads the language model. Later runs are quick |
| Adding texts is slow | A model reads every text. About 6 texts a second on the author's laptop CPU, so half an hour for 10,000. It is done once: save, and open next time |
| Many texts, much memory | Meant for up to some tens of thousands of texts (to about 100,000). The texts are all held in memory |
| Japanese texts | `TextIndex("ja", sparse="bm25")` |
| Reworded questions miss | Try "Two models" below |

## Two models (the default)

`sparse="bm25"` in the examples is the quick setting: one model for the meaning, and matching of the written words.
Without it, the word side uses a model too.

```python
ix = TextIndex("en")          # for "ja" this also needs: pip install fugashi unidic-lite
```

| | `sparse="bm25"` | default |
|---|---|---|
| Models downloaded | one | two |
| Adding texts | quick (about 10 times in the author's measurement) | slow |
| Good for | questions in the texts' own words; names, part numbers, abbreviations | short, reworded questions |

In the author's measurement on 637 Japanese journal papers, short questions did slightly better with the default
(0.970 against 0.959) and long ones with `sparse="bm25"` (0.983 against 0.991). It depends on the data: when in
doubt, start with the quick one.

## Good to know

- **Experimental.** Names, arguments and the saved form of `kci_vec.text` may change.
- **The language models belong to others.** Built in: `intfloat/multilingual-e5-small` (MIT) and
  `opensearch-project/opensearch-neural-sparse-encoding-v2-distill` (Apache-2.0) for English; `cl-nagoya/ruri-v3-30m`
  (Apache-2.0) and `hotchpotch/japanese-splade-v2` (MIT) for Japanese (as their model pages state; check for yourself).
- **KCI-Vec's own licence**: see `LICENSE` in the repository (source-available for non-commercial use; commercial use
  under a separate agreement).
- It does not write answers (no generation). You get the texts that fit, and the best part of each.
