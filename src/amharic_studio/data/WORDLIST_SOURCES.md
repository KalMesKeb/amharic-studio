# Where `amharic_wordlist.txt.gz` comes from

The bundled wordlist is a frequency table of Amharic surface forms, produced by
`tools/build_lexicon.py`. It contains only words and the number of times each was seen;
no sentence, phrase or other running text from any source is reproduced.

It exists because Amharic is heavily affixed. A curated list of citation forms leaves a
spell-checker flagging most of the verbs on a page, which makes the review queue useless.
Counting surface forms in real text is the only practical way to get coverage offline.

## Sources

| Source | What it is | Licence |
| --- | --- | --- |
| [Amharic Wikipedia](https://dumps.wikimedia.org/amwiki/) | Article dump, edited prose | CC BY-SA 4.0 |
| [MADLAD-400](https://huggingface.co/datasets/allenai/madlad-400) (`am`, cleaned) | Audited web crawl, AllenAI | ODC-BY 1.0 |

`tools/build_lexicon.py` can also read [CC-100](https://data.statmt.org/cc-100/)
Amharic, which is larger again, but statmt.org serves it slowly enough that it is not
used for the committed list.

## Rebuilding

```
python tools/build_lexicon.py --source wikipedia madlad --min-freq 3
```

Both sources are cached under `scratch/corpora/`, so a rebuild after the first run only
costs the tokenisation. Raising `--min-freq` trims the tail of typos and OCR noise that
any crawl carries; lowering it improves coverage of rare but real vocabulary at the cost
of also recognising misspellings.

## What this list is not

It is a recognition aid, not an authority on spelling. Words are recorded as they were
written, so both `ሰላሳ` and `ሠላሳ` are present, and so are forms a careful editor would
reject. Deciding between spellings is the job of the normalisation policy, and offering
corrections is the job of the confusion model — this file only answers "has anyone
written this before, and how often".
