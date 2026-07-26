# Amharic Studio

An offline desktop program for turning badly OCR'd Amharic books into clean, correct,
readable editions — searchable PDFs that still look like the original scan, and typeset
PDFs and EPUBs that do not.

Everything runs on your machine. No account, no network calls, no telemetry, no cloud
OCR. Once Tesseract and a font are installed you can unplug the network and the program
works exactly the same.

---

## Why this is not a general OCR tool

Amharic is written in the Ge'ez syllabary: around 280 characters arranged as consonant
families in seven or eight vowel orders. Within a family the glyphs differ by a small
appendage — ተ ቱ ቲ ታ ቴ ት ቶ are one consonant with seven vowels — so the errors a
recognizer makes on a worn page are overwhelmingly *right consonant, wrong vowel*. A
spellchecker built for Latin treats those as arbitrary substitutions and ranks the true
reading below a dozen unrelated words.

The whole program is arranged around that fact:

- **Retrieval by consonant skeleton.** Strip the vowel orders off a word and the skeleton
  usually survives the error intact, so one indexed lookup finds the intended word where
  generic edit distance would have to scan a huge neighbourhood.
- **Confusion-weighted distance.** A vowel-order slip inside one family costs far less
  than a jump to an unrelated letter, and the weights *learn from the corrections you
  actually make* as you work through a book.
- **Morphological awareness.** Amharic glues prepositions, articles and case markers onto
  stems: ኢትዮጵያ, የኢትዮጵያ, በኢትዮጵያን, ኢትዮጵያውያን. A plain wordlist reports most correct text as
  unknown, so affixed forms are analysed back to their stems before anything is flagged.
- **A dictionary counted from real text.** Affixation also defeats hand-written wordlists:
  one verb has hundreds of surface forms, and a checker that knows only the citation form
  flags most of the verbs on a page. The bundled dictionary is a frequency table built
  from public Amharic corpora, so ordinary inflection is simply *known*.
- **Old and new orthography, separately.** ሠ/ሰ, ኀ/ሀ, ዐ/አ and ፀ/ጸ are the same sounds
  spelled differently across eras. The book's own spelling is preserved exactly as
  printed; regularization is a *view and export option*, never a silent rewrite.

## What it does

**Import** scans, photographs, or existing PDFs. A PDF that already has a text layer can
be repaired without re-recognizing it.

**Preprocess** each page: deskew, binarize with a local (Sauvola) threshold that survives
uneven photocopier shading, despeckle, and clear the black scanner gutter — without
moving the page, so word boxes stay aligned to the image.

**Recognize** with Tesseract, optionally running more than one language model over the
same page. Where two models disagree, that disagreement is surfaced as a review flag:
an engine can be confidently wrong, but two rarely fail the same way.

**Correct** side by side. The scan sits next to the editable text, scrolling together,
with a confidence heat map over the word boxes and inline marks on everything suspect.
Every flag comes with candidate readings, each with a plain-language reason. Type in
fidel directly, or phonetically in Latin (`selam` → ሰላም) if you have no Amharic keyboard.

**Triage** with a quality dashboard that ranks pages worst-first, so you spend your time
where the text is actually broken instead of paging through a book in order.

**Export** to searchable PDF (original scan, invisible corrected text layer), typeset PDF,
EPUB 3, DOCX-friendly Markdown, plain text, and ALTO XML or hOCR for interoperability with
other ground-truthing tools. Ethiopic fonts are subsetted and embedded, and the language
is declared as `am` so the file renders correctly wherever it is opened.

### Nothing is corrected behind your back

A wrong "fix" applied silently is worse than the OCR error it replaced, because you will
never see it again. So a suggestion is only ever applied automatically when it is
*verified* — a real lexicon word or a deterministic Unicode normalization — and never when
the word it would replace is already a word. Everything else waits for you. Every change,
automatic or manual, is recorded in a per-book ledger.

---

## Installation

### 1. Python

Python 3.10 or newer.

```bash
git clone <this repository>
cd amharic-studio
python -m venv .venv
```

```bash
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e ".[imaging]"   # OpenCV + SciPy: faster preprocessing and despeckling
pip install -e ".[dev]"       # pytest and ruff
```

The program runs without the extras; they only make preprocessing faster.

### 2. Tesseract and the Amharic language data

Tesseract is a separate program and must be installed through your system, not pip.

**Windows** — install the [UB Mannheim build](https://github.com/UB-Mannheim/tesseract/wiki)
and tick **Amharic** under *Additional language data* during setup. If Tesseract is not on
your `PATH`, set `TESSERACT_CMD` to the executable:

```powershell
$env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

**macOS**

```bash
brew install tesseract tesseract-lang
```

**Debian / Ubuntu**

```bash
sudo apt install tesseract-ocr tesseract-ocr-amh tesseract-ocr-script-ethi
```

**Fedora**

```bash
sudo dnf install tesseract tesseract-langpack-amh
```

Two models are worth having: `amh` (Amharic) and `script/Ethiopic` (the script model,
trained more broadly). Running both and letting them vote is the single cheapest accuracy
improvement available, and their disagreements make an excellent review queue.

If you want to install the language data by hand, download `amh.traineddata` and
`script/Ethiopic.traineddata` from
[tessdata_best](https://github.com/tesseract-ocr/tessdata_best) and drop them in your
`tessdata` directory. `tessdata_best` is noticeably more accurate than `tessdata_fast` on
degraded scans and is worth the extra runtime.

### 3. An Ethiopic font

You need one font that covers the Ge'ez block, both for the editor and for embedding in
exports. Most systems already have one:

| Platform | Usually already installed |
|---|---|
| Windows | Nyala, Ebrima |
| macOS | Kefa |
| Linux | often none — install one |

The best free option is [Abyssinica SIL](https://software.sil.org/abyssinica/) (SIL Open
Font License). On Debian/Ubuntu: `sudo apt install fonts-sil-abyssinica`.

### 4. Check the installation

```bash
amharic-studio info
```

This prints every engine, language model, font and optional component it can find, and
says plainly what is missing. Start here whenever something is not working.

### 5. The dictionary (already included)

A frequency list of Amharic surface forms ships inside the package, so nothing needs to
be downloaded. It is loaded into your user lexicon the first time the program starts,
which takes a few seconds and shows a splash while it happens; every later start is
instant.

If you are working from a source checkout that does not have it, or you want to rebuild
it from newer corpus dumps:

```bash
python tools/build_lexicon.py --source wikipedia madlad --min-freq 3
```

Sources and licences are listed in `src/amharic_studio/data/WORDLIST_SOURCES.md`. Without
the file the program still runs, falling back to a small curated seed list — but it will
flag a great deal of perfectly good Amharic.

You can also teach it the vocabulary of your own field, which is worth doing before a
long book in a specialised register:

```bash
amharic-studio lexicon my-corpus/*.txt
```

---

## Running it

```bash
amharic-studio            # open the window
```

The engine also runs headless, which is how you process a shelf of books unattended:

```bash
amharic-studio ocr    "My Book.amproj" --engine tesseract:amh --engine tesseract:script/Ethiopic
amharic-studio ocr    "My Book.amproj" --auto-correct
amharic-studio check  "My Book.amproj"              # quality report, worst pages first
amharic-studio export "My Book.amproj" --format epub --format pdf
amharic-studio lexicon corpus/*.txt                 # teach it vocabulary from your own texts
```

### A first pass through a book

1. **File → New project**, then **Import pages** and choose your scans or PDF.
2. **Recognize.** Pick both language models if you have them.
3. Open the **Quality** tab and work the list from the top: it is sorted worst-first.
4. On each page, `F3` jumps to the next flag. `Enter` accepts the top suggestion, `Esc`
   dismisses it, and typing just edits the text.
5. Mark a page **Verified** when you are happy with it. Verified pages become training
   data and feed the confusion model, so accuracy improves as you go.
6. **Export** when the quality dashboard looks the way you want it.

The keyboard map is under **Help → Keyboard shortcuts**. The workflow is built to be
driven almost entirely from the keyboard, because correcting a book means doing the same
three keystrokes several thousand times.

---

## Where things are kept

A project is a `.amproj` directory: a SQLite database plus the page images. It is a plain
directory, so you can copy, sync or version it. The database holds the original OCR, your
edited text, word geometry, structure marks and the full edit ledger — the original
recognizer output is never overwritten, which is what makes the side-by-side comparison
and the training export possible.

The lexicon and the learned confusion model live in your user data directory
(`amharic-studio info` prints the path) and are shared across books. Deleting the lexicon
is safe: it is rebuilt from the bundled wordlist on the next start, though any vocabulary
you added yourself goes with it.

---

## Development

```bash
pip install -e ".[dev]"
pytest                    # 380 tests; Qt and Tesseract ones skip if absent
ruff check src tests
```

The core is deliberately free of Qt imports, so all of the script handling, correction,
storage and export logic is testable headlessly and reusable from a script.

```
src/amharic_studio/
  core/
    fidel.py        the syllabary: families, orders, folding, punctuation, numerals
    morph.py        affix stripping, so inflections are not mistaken for errors
    normalize.py    Unicode and orthography policies, every change reported
    confusion.py    visually-weighted edit distance that learns from your corrections
    lexicon.py      SQLite word store, skeleton and deletion indexes
    suggest.py      issue detection and candidate ranking
    imaging.py      deskew, binarize, despeckle, border removal
    ocr/            pluggable backends, Tesseract, multi-engine voting
    export/         EPUB, typeset PDF, searchable PDF, ALTO, hOCR, text
    project.py      the .amproj container
    qa.py           page and book quality metrics
    pipeline.py     recognition and correction orchestration
  data/             the seed list and the bundled corpus wordlist
  ui/               PySide6 window, canvas, editor, panels, workers
tools/
  build_lexicon.py  rebuilds the bundled wordlist from public corpora
```

The two SQLite-backed stores, `Project` and `Lexicon`, are read and written from worker
threads while the UI keeps using them, so both are safe to share: the project hands each
thread its own connection over a WAL database, and the lexicon serialises on a lock.

Adding another recognizer means implementing `OcrBackend` and registering it; nothing else
changes.

## License

MIT. Tesseract, Abyssinica SIL and the other components have their own licenses.
