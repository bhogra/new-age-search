// Search worker: downloads the index once (kept in the browser's cache), then
// runs all searching locally — nothing is sent to any server.
const TRANSFORMERS_SRC = self.TRANSFORMERS_SRC ?? "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";
const MODEL = "Xenova/bge-small-en-v1.5";
const PREFIX = "Represent this sentence for searching relevant passages: ";
const DIM = 384;
const SEP = "\x1f";

let meta, emb, scales, blob, offsets, vocab, embedder;

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === "load") await load();
    else if (m.type === "search") await search(m);
    else if (m.type === "article") article(m);
  } catch (err) {
    postMessage({ type: "error", message: String(err.stack || err) });
  }
};

const progress = (frac, label) => postMessage({ type: "progress", frac, label });
const MB = (n) => Math.round(n / 1e6);

async function fetchCached(url, onBytes) {
  const cache = typeof caches === "undefined" ? null : await caches.open("new-age-data-v1");
  const hit = cache && await cache.match(url);
  if (hit) {
    const buf = new Uint8Array(await hit.arrayBuffer());
    onBytes(buf.length);
    return buf;
  }
  const parts = [];
  let len = 0;
  const reader = (await fetch(url)).body.getReader();
  for (let r; !(r = await reader.read()).done; ) {
    parts.push(r.value);
    len += r.value.length;
    onBytes(r.value.length);
  }
  const buf = new Uint8Array(len);
  let o = 0;
  for (const p of parts) { buf.set(p, o); o += p.length; }
  if (cache) await cache.put(url, new Response(buf));
  return buf;
}

const gunzip = async (buf) =>
  new Response(new Response(buf).body.pipeThrough(new DecompressionStream("gzip"))).text();

async function load() {
  progress(0, "waking up…");
  meta = JSON.parse(await gunzip(await fetchCached("data/meta.json.gz", () => {})));
  const total = Object.values(meta.bytes).reduce((a, b) => a + b, 0);
  let got = 0;
  const tick = (n) => {
    got += n;
    progress(0.72 * got / total, `downloading the archive — ${MB(got)} of ${MB(total)} MB`);
  };
  const files = {};
  await Promise.all(Object.keys(meta.bytes).map(async (f) => { files[f] = await fetchCached("data/" + f, tick); }));

  progress(0.73, "unpacking the pages…");
  emb = new Int8Array(meta.n_chunks * DIM);
  for (let i = 0; files[`emb_${i}.bin`]; i++) emb.set(files[`emb_${i}.bin`], i * meta.rows_per_shard * DIM);
  scales = new Float32Array(files["emb_scales.bin"].buffer);
  const texts = [];
  for (let i = 0; files[`text_${i}.gz`]; i++) texts.push(await gunzip(files[`text_${i}.gz`]));
  blob = texts.join(SEP);
  offsets = new Uint32Array(meta.n_chunks + 1);
  for (let c = 1, p = 0; c <= meta.n_chunks; c++)
    offsets[c] = p = (c < meta.n_chunks ? blob.indexOf(SEP, p) : blob.length) + 1;
  vocab = (await gunzip(files["vocab.txt.gz"])).split("\n").filter(Boolean);

  const { pipeline } = await import(TRANSFORMERS_SRC);
  const loaded = {};
  const pipe = await pipeline("feature-extraction", MODEL, {
    dtype: "q8",
    progress_callback: (p) => {
      if (p.status !== "progress") return;
      loaded[p.file] = p.loaded;
      const sum = Object.values(loaded).reduce((a, b) => a + b, 0);
      progress(0.75 + 0.23 * Math.min(1, sum / 35e6), `downloading the language model — ${MB(sum)} of 35 MB`);
    },
  });
  embedder = async (q) => (await pipe([PREFIX + q], { pooling: "cls", normalize: true })).data;
  progress(0.99, "warming up…");
  await embedder("warmup");
  postMessage({ type: "ready" });
}

// ---------- shared lookups ----------

// first index where arr[i] > x
function upperBound(arr, x) {
  let lo = 0, hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid] > x) hi = mid; else lo = mid + 1;
  }
  return lo;
}

const chunkText = (cid) => blob.slice(offsets[cid], offsets[cid + 1] - 1);
const artOf = (cid) => upperBound(meta.art_start, cid) - 1;

function rowFor(cid, extra) {
  const a = artOf(cid);
  const iss = meta.art_issue[a];
  return {
    chunk_id: cid, art: a,
    issue_id: meta.issues[iss], date: meta.issue_dates[iss],
    heading: meta.headings[meta.art_head[a]], article_num: meta.art_num[a],
    para_idx: cid - meta.art_start[a], text: chunkText(cid), ...extra,
  };
}

// ---------- search by meaning ----------

async function semanticList(q, pool) {
  const qf = await embedder(q);
  const n = meta.n_chunks;
  const scores = new Float32Array(n);
  for (let i = 0, off = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < DIM; j++) s += emb[off++] * qf[j];
    scores[i] = s * scales[i];
  }
  const idx = new Uint32Array(n);
  for (let i = 0; i < n; i++) idx[i] = i;
  idx.sort((a, b) => scores[b] - scores[a]);
  return Array.from(idx.slice(0, pool), (cid) => ({ cid, score: Math.round(scores[cid] * 1e4) / 1e4 }));
}

// ---------- search by spelling ----------
// Same scoring as rapidfuzz fuzz.ratio: 200*LCS/(len_a+len_b), via bit-parallel LCS.

function charMasks(a) {
  const m = new Map();
  for (let i = 0; i < a.length; i++) m.set(a.charCodeAt(i), (m.get(a.charCodeAt(i)) ?? 0) | (1 << i));
  return m;
}

function lcsDP(a, b) {
  let prev = new Uint16Array(b.length + 1), cur = new Uint16Array(b.length + 1);
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < b.length; j++)
      cur[j + 1] = a[i] === b[j] ? prev[j] + 1 : Math.max(prev[j + 1], cur[j]);
    [prev, cur] = [cur, prev];
  }
  return prev[b.length];
}

function popcount(x) {
  x -= (x >> 1) & 0x55555555;
  x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
  return (((x + (x >> 4)) & 0x0f0f0f0f) * 0x01010101) >> 24;
}

function ratio(a, masks, b, cutoff) {
  const la = a.length, lb = b.length;
  if (200 * Math.min(la, lb) / (la + lb) < cutoff) return 0;
  let lcs;
  if (la <= 30) {
    const mask = (1 << la) - 1;
    let S = mask;
    for (let i = 0; i < lb; i++) {
      const u = S & (masks.get(b.charCodeAt(i)) ?? 0);
      S = ((S + u) | (S - u)) & mask;
    }
    lcs = popcount(~S & mask);
  } else lcs = lcsDP(a, b);
  return 200 * lcs / (la + lb);
}

function corpusVariants(word, cutoff) {
  if (word.length < 3) return [[word, 100]]; // too short for the variant dictionary — match as-is
  const masks = charMasks(word);
  const out = [];
  for (const w of vocab) {
    const s = ratio(word, masks, w, cutoff);
    if (s >= cutoff) out.push([w, s]);
  }
  out.sort((x, y) => y[1] - x[1]);
  return out.slice(0, 25);
}

const reEsc = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

function scanBlob(pattern, scoreOf) {
  const hits = new Map(); // cid -> {variant, score, count}
  for (let m; (m = pattern.exec(blob)); ) {
    const cid = upperBound(offsets, m.index) - 1;
    const score = scoreOf(m);
    const h = hits.get(cid);
    if (!h) hits.set(cid, { variant: m[0], score, count: 1 });
    else {
      h.count++;
      if (score > h.score) { h.score = score; h.variant = m[0]; }
    }
  }
  return [...hits.entries()]
    .map(([cid, h]) => ({ cid, score: Math.round(h.score * 10) / 10, matched_variant: h.variant, count: h.count }))
    .sort((a, b) => b.score - a.score || b.count - a.count);
}

function fuzzyList(q, cutoff) {
  const words = q.toLowerCase().split(/\W+/).filter(Boolean);
  const perWord = words.map((w) => corpusVariants(w, cutoff));
  if (perWord.some((v) => !v.length)) return { list: [], variants: [] };
  if (words.length === 1) {
    const vscore = new Map(perWord[0]);
    const pat = new RegExp("\\b(?:" + perWord[0].map(([t]) => reEsc(t)).join("|") + ")\\b", "gi");
    return { list: scanBlob(pat, (m) => vscore.get(m[0].toLowerCase())), variants: perWord[0] };
  }
  // phrase: each word may appear as any of its variant spellings, words adjacent.
  // [^\w\x1f] keeps a match inside one passage (\x1f separates passages).
  const maps = perWord.map((vs) => new Map(vs));
  const pat = new RegExp(
    "\\b(" + perWord.map((vs) => vs.map(([t]) => reEsc(t)).join("|")).join(")[^\\w\\x1f]+(") + ")\\b", "gi");
  const scoreOf = (m) => maps.reduce((s, mp, i) => s + mp.get(m[i + 1].toLowerCase()), 0) / maps.length;
  return { list: scanBlob(pat, scoreOf), variants: perWord.flat() };
}

// ---------- combined + article grouping ----------

async function search({ q, mode, topK, cutoff }) {
  const t0 = performance.now();
  let list, variants = [], fuzzy = null;
  if (mode === "meaning") {
    list = await semanticList(q, 300);
  } else if (mode === "spelling") {
    ({ list, variants } = fuzzyList(q, cutoff));
    fuzzy = list;
  } else {
    const sem = await semanticList(q, 100);
    ({ list: fuzzy, variants } = fuzzyList(q, cutoff));
    const fused = new Map(); // reciprocal rank fusion, as in the original app
    for (const [name, results] of [["meaning", sem], ["spelling", fuzzy.slice(0, 100)]])
      results.forEach((r, rank) => {
        const e = fused.get(r.cid) ?? { cid: r.cid, rrf: 0, modes: [] };
        e.rrf += 1 / (60 + rank + 1);
        e.modes.push(name);
        Object.assign(e, r);
        fused.set(r.cid, e);
      });
    list = [...fused.values()].sort((a, b) => b.rrf - a.rrf)
      .map((e) => ({ ...e, rrf_score: Math.round(e.rrf * 1e4) / 1e4, found_by: e.modes.join("+") }));
  }

  // one row per article: best passage is the snippet, occurrences summed across the article
  const artHits = new Map();
  if (fuzzy) for (const e of fuzzy) {
    const a = artOf(e.cid);
    artHits.set(a, (artHits.get(a) ?? 0) + e.count);
  }
  const arts = new Map();
  for (const { cid, rrf, modes, count, ...fields } of list) {
    const a = artOf(cid);
    const r = arts.get(a);
    if (r) r.hits += artHits.has(a) ? 0 : 1;
    else arts.set(a, rowFor(cid, { ...fields, hits: artHits.get(a) ?? 1 }));
  }
  const results = [...arts.values()].slice(0, topK || undefined);
  results.forEach((r, i) => { r.rank = i; });
  postMessage({
    type: "results", results,
    variants: variants.map(([token, score]) => ({ token, score })),
    took: Math.round((performance.now() - t0)) / 1000,
  });
}

export { ratio, charMasks, corpusVariants }; // for the test harness; unused by the browser

function article({ art }) {
  const end = meta.art_start[art + 1] ?? meta.n_chunks;
  const texts = [];
  for (let c = meta.art_start[art]; c < end; c++) texts.push(chunkText(c));
  postMessage({
    type: "article",
    issue_id: meta.issues[meta.art_issue[art]],
    heading: meta.headings[meta.art_head[art]],
    text: texts.join("\n\n"),
  });
}
