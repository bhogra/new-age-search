# The New Age — Keyword Searcher

Search every page of *The New Age* (1907–1922): 16,064 articles across 806 issues,
digitised from the original printed volumes.

**Live site:** https://bhogra.github.io/new-age-search/

Everything runs in your browser. The first visit downloads the full archive and a small
language model (about 160 MB, kept by your browser for next time); after that, searches
run instantly on your own machine and never touch a server.

## How searching works

- **meaning** — finds passages about your idea, even when the exact word never appears.
- **spelling** — finds a word plus its variant spellings (the original printing introduced
  many errors, e.g. *feinism* for *feminism*, *sufferage* for *suffrage*). The "spelling match" number controls how
  loose this is: lower catches more printing errors, 100 means exact spelling only.
- **meaning + spelling** (default) — both, merged.

Each article appears once in the results, with its best-matching passage as the snippet
and a count of every hit inside the article. Results can be sorted by best match, most
hits, or publication date (dates are best estimates for issues whose printed date is
missing or implausible). Click a heading to read the whole article with all hits
highlighted. Export CSV downloads whatever is currently in the results window.

## Repository layout

- `index.html` — the page (Windows-95 style, no framework, no build step)
- `worker.js` — a web worker that downloads the index and runs all three searches
- `data/` — the prebuilt index: passage embeddings (int8), passage text (gzip),
  spelling dictionary, article/issue/date tables
- `build.py` — rebuilds `data/` from the research project's index
  (`chunks.parquet`, `embeddings.npy`, `vocab.txt`, `issue_dates.csv`)
- `compare.py` — checks that the compact int8 index ranks passages like the original

## Rebuilding the data

```
python build.py --src /path/to/keyword-searcher/index --out data
python compare.py --src /path/to/keyword-searcher/index   # expect ≥8/10 top-10 overlap
```

## Notes on fidelity

The site reproduces the original research tool (a local Python app) with small,
deliberate differences:

- Multi-word searches in spelling mode match each word's variant spellings with the
  words adjacent, rather than scanning for arbitrary near-matches of the whole phrase.
- Typographic punctuation (curly quotes, dashes) is shown as plain ASCII.
- Passage embeddings are stored at int8 precision and the query model is quantised;
  top-10 rankings match the originals at 9–10 out of 10.

## Local development

```
python3 -m http.server 8000   # then open http://localhost:8000
```
