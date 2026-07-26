"""Build the bundled Amharic wordlist from a public corpus.

Amharic is heavily affixed, so a hand-written wordlist is hopeless: ``ኖረ`` (to live)
alone has hundreds of surface forms, and a spell-checker that only knows the citation
form flags almost every verb in a real book. The fix is to count surface forms in real
text, which is what this script does. Corrections in the app then have something to
aim at, and the QA "unknown words" figure means something.

This runs once, offline of the app, and its output is committed:

    python tools/build_lexicon.py --source wikipedia            # ~9 MB, about a minute
    python tools/build_lexicon.py --source wikipedia madlad     # what is committed
    python tools/build_lexicon.py --source cc100 --min-freq 5   # if you have the patience

Sources
    wikipedia   The Amharic Wikipedia article dump. Small, clean, edited prose, and
                CC-BY-SA.
    madlad      MADLAD-400's cleaned Amharic web documents, 33 shards of about 2 MB,
                served from a CDN that is actually fast. Noisier than Wikipedia but
                broad enough to cover everyday inflection, which is the whole point.
    cc100       CC-100 Amharic. Larger still, but statmt.org serves it at a few KB/s,
                so it is here for completeness rather than for regular use.

Attribution for whatever ends up bundled lives in ``data/WORDLIST_SOURCES.md``.

A frequency list is a set of measurements about a corpus, not a copy of it; only the
counts and the surface forms are kept, and nothing longer than a word ever survives.
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import json
import lzma
import re
import sys
import time
import unicodedata
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "src" / "amharic_studio" / "data" / "amharic_wordlist.txt.gz"
CACHE = ROOT / "scratch" / "corpora"

# Wikimedia rejects anonymous bulk clients, and rightly so.
USER_AGENT = "amharic-studio-lexicon-builder/0.1 (offline Amharic OCR editor)"

SOURCES = {
    "wikipedia": "https://dumps.wikimedia.org/amwiki/latest/amwiki-latest-pages-articles.xml.bz2",
    "cc100": "https://data.statmt.org/cc-100/am.txt.xz",
}

MADLAD_REPO = "allenai/madlad-400"
MADLAD_DIR = "data-v1p5/am"

#: A word is a run of Ethiopic letters. Combining marks and the Ethiopic digits are
#: excluded: digits are not vocabulary, and a stray mark means the token is damaged.
WORD_RE = re.compile(r"[\u1200-\u135A\u1380-\u138F\u2D80-\u2DDE\uAB01-\uAB2E]+")

#: Longest plausible Amharic surface form. Anything past this is glued-together OCR
#: noise or a URL fragment that survived cleaning, and it only pollutes the index.
MAX_LEN = 24
MIN_LEN = 2

#: Wiki markup that would otherwise contribute template and parameter names as "words".
WIKI_NOISE = re.compile(
    r"""
      <ref[^>]*>.*?</ref>      # citations
    | <[^>]+>                  # any other tag
    | \{\{[^{}]*\}\}           # templates (one nesting level, applied repeatedly)
    | \[\[[^\]|]*\|            # piped link targets, keeping the display text
    | https?://\S+
    | \[https?://[^\]]*\]
    """,
    re.VERBOSE | re.DOTALL,
)


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def download(url: str, target: Path, timeout: float = 60.0) -> Path:
    """Fetch ``url`` to ``target``, reusing an existing complete download."""
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        total = int(response.headers.get("Content-Length", 0))
        if target.exists() and total and target.stat().st_size == total:
            log(f"  cached {target.name} ({total / 1e6:.1f} MB)")
            return target

        log(f"  downloading {url} ({total / 1e6:.1f} MB)")
        partial = target.with_suffix(target.suffix + ".part")
        done = 0
        last = 0.0
        with partial.open("wb") as fh:
            while chunk := response.read(1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if total and time.monotonic() - last > 2:
                    log(f"    {done / 1e6:6.1f} / {total / 1e6:.1f} MB")
                    last = time.monotonic()
                # These servers do not always close the connection after the last byte,
                # and a read that waits for an EOF that never arrives hangs the build.
                if total and done >= total:
                    break
        partial.replace(target)
    return target


def strip_wiki(markup: str) -> str:
    text = markup
    for _ in range(3):  # templates nest; a few passes flattens the common cases
        text, n = WIKI_NOISE.subn(" ", text)
        if not n:
            break
    return text


def wikipedia_texts(path: Path):
    """Yield the wikitext of every article, streaming so the dump never lands in RAM."""
    with bz2.open(path, "rb") as raw:
        for _event, element in ET.iterparse(raw, events=("end",)):
            tag = element.tag.rpartition("}")[2]
            if tag == "text":
                if element.text:
                    yield strip_wiki(element.text)
                element.clear()
            elif tag == "page":
                element.clear()


def cc100_texts(path: Path):
    with lzma.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        yield from fh


def madlad_shards() -> list[str]:
    """Resolve the Amharic shard URLs from the dataset listing."""
    api = (
        f"https://huggingface.co/api/datasets/{MADLAD_REPO}"
        f"/tree/main/{MADLAD_DIR}?recursive=1&expand=1"
    )
    request = urllib.request.Request(api, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        listing = json.load(response)
    # The directory also holds ``noisy_docs`` shards, which are the documents MADLAD's
    # own audit rejected. Counting them would put back exactly the crawl garbage the
    # cleaned split exists to keep out.
    paths = sorted(
        f["path"]
        for f in listing
        if f["path"].endswith(".jsonl.gz") and "clean_docs" in f["path"]
    )
    return [
        f"https://huggingface.co/datasets/{MADLAD_REPO}/resolve/main/{p}" for p in paths
    ]


def madlad_texts(limit_shards: int = 0):
    """Yield document text from the cleaned Amharic shards."""
    urls = madlad_shards()
    if limit_shards:
        urls = urls[:limit_shards]
    log(f"  {len(urls)} shards")
    for url in urls:
        path = download(url, CACHE / "madlad" / url.rpartition("/")[2])
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    yield json.loads(line).get("text", "")
                except json.JSONDecodeError:
                    continue


def count(texts, counter: Counter, label: str) -> None:
    """Accumulate surface-form frequencies, normalising to NFC as we go."""
    docs = 0
    start = time.monotonic()
    last = start
    for text in texts:
        docs += 1
        for match in WORD_RE.finditer(unicodedata.normalize("NFC", text)):
            word = match.group()
            if MIN_LEN <= len(word) <= MAX_LEN:
                counter[word] += 1
        if time.monotonic() - last > 5:
            log(f"    {label}: {docs:,} docs, {len(counter):,} types")
            last = time.monotonic()
    log(f"  {label}: {docs:,} docs, {len(counter):,} types in {time.monotonic() - start:.0f}s")


def write(counter: Counter, out: Path, min_freq: int, limit: int) -> int:
    """Write ``word<TAB>freq``, most frequent first, gzipped."""
    kept = [(w, c) for w, c in counter.items() if c >= min_freq]
    kept.sort(key=lambda wc: (-wc[1], wc[0]))
    if limit:
        kept = kept[:limit]

    out.parent.mkdir(parents=True, exist_ok=True)
    # mtime=0 so rebuilding an unchanged list produces an identical file and does not
    # show up as a spurious diff.
    with gzip.GzipFile(out, "wb", compresslevel=9, mtime=0) as gz:
        gz.write(f"# amharic surface forms, word<TAB>frequency, min_freq={min_freq}\n".encode())
        for word, freq in kept:
            gz.write(f"{word}\t{freq}\n".encode())
    return len(kept)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    choices = sorted([*SOURCES, "madlad"])
    parser.add_argument("--source", nargs="+", choices=choices, default=["wikipedia", "madlad"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--min-freq",
        type=int,
        default=2,
        help="drop words seen fewer times; 1 keeps the long tail of typos and OCR noise",
    )
    parser.add_argument("--limit", type=int, default=0, help="keep only the N most frequent")
    parser.add_argument("--shards", type=int, default=0, help="madlad only: stop after N shards")
    args = parser.parse_args(argv)

    counter: Counter[str] = Counter()
    for name in args.source:
        log(f"{name}:")
        if name == "madlad":
            count(madlad_texts(args.shards), counter, name)
            continue
        url = SOURCES[name]
        path = download(url, CACHE / url.rpartition("/")[2])
        reader = wikipedia_texts if name == "wikipedia" else cc100_texts
        count(reader(path), counter, name)

    written = write(counter, args.out, args.min_freq, args.limit)
    size = args.out.stat().st_size
    log(f"wrote {written:,} words to {args.out} ({size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
