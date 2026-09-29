#!/usr/bin/env python3
"""
Stanza Revisited: extended experiments for the poster.

Adds to the original run_experiments.py:
  1. UPOS / UAS / LAS for every model x treebank (UAS was missing from Table 1)
  2. Paired bootstrap significance tests (pooled vs. single-treebank models),
     per test set and for the held-out mean  -> replaces "single run, no significance test"
  3. Per-relation LAS-F1 and LAS by sentence length / dependency distance (error analysis)
  4. Additional languages (German, French) with the same trained-vs-held-out design
  5. Speed: N timed runs (median + IQR), tokens/s, CPU and GPU if available,
     spaCy sm and (optionally) trf as baselines

Everything (numbers, versions, hardware, resolved model paths) goes to
results_extended.json; predictions go to preds/ as CoNLL-U.

Usage
  pip install stanza spacy numpy
  python -m spacy download en_core_web_sm        # (+ en_core_web_trf for --spacy-trf)
  python run_extended.py                          # accuracy + CPU speed, all languages
  python run_extended.py --langs en               # English only
  python run_extended.py --speed-only --device gpu  # e.g. on Google Colab (T4)
"""
import argparse, json, os, platform, statistics, sys, time
from collections import Counter, defaultdict
from pathlib import Path

UD_TAG_DEFAULT = "r2.15"

# lang -> {"models": {label: stanza package}, "tests": {name: (repo, file)},
#          "trained_on": {label: [test names the model was trained on]}}
# "default" is Stanza's pooled ("combined") package where one exists.
CONFIG = {
    "en": {
        "models": {"ewt": "ewt", "gum": "gum", "combined": "default"},
        "tests": {
            "EWT": ("UD_English-EWT", "en_ewt-ud-test.conllu"),
            "GUM": ("UD_English-GUM", "en_gum-ud-test.conllu"),
            "LinES": ("UD_English-LinES", "en_lines-ud-test.conllu"),
            "ParTUT": ("UD_English-ParTUT", "en_partut-ud-test.conllu"),
        },
        "trained_on": {"ewt": ["EWT"], "gum": ["GUM"], "combined": ["EWT", "GUM"]},
        "pooled": "combined",
    },
    "de": {
        "models": {"gsd": "gsd", "hdt": "hdt", "combined": "default"},
        "tests": {
            "GSD": ("UD_German-GSD", "de_gsd-ud-test.conllu"),
            "HDT": ("UD_German-HDT", "de_hdt-ud-test.conllu"),
            "PUD": ("UD_German-PUD", "de_pud-ud-test.conllu"),
        },
        "trained_on": {"gsd": ["GSD"], "hdt": ["HDT"], "combined": ["GSD", "HDT"]},
        "pooled": "combined",
    },
    "fr": {
        "models": {"gsd": "gsd", "sequoia": "sequoia", "combined": "default"},
        "tests": {
            "GSD": ("UD_French-GSD", "fr_gsd-ud-test.conllu"),
            "Sequoia": ("UD_French-Sequoia", "fr_sequoia-ud-test.conllu"),
            "ParTUT": ("UD_French-ParTUT", "fr_partut-ud-test.conllu"),
            "PUD": ("UD_French-PUD", "fr_pud-ud-test.conllu"),
        },
        "trained_on": {"gsd": ["GSD"], "sequoia": ["Sequoia"], "combined": ["GSD", "Sequoia"]},
        "pooled": "combined",
    },
}
# NOTE: which treebanks Stanza's "default"/combined package was trained on changes
# between releases. Check the Stanza model docs for your version and fix
# "trained_on" above before reading held-out results. The script records the
# resolved model file names so this can be verified afterwards.

PROCESSORS = "tokenize,pos,lemma,depparse"


# ----------------------------------------------------------------- CoNLL-U
def read_conllu(path):
    """Syntactic words only (skips multiword ranges and empty nodes)."""
    sents, cur, meta = [], [], []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            if cur:
                sents.append({"meta": meta, "words": cur})
            cur, meta = [], []
        elif line.startswith("#"):
            meta.append(line)
        else:
            c = line.split("\t")
            if "-" in c[0] or "." in c[0]:
                continue
            cur.append({"form": c[1], "upos": c[3], "head": int(c[6]),
                        "deprel": c[7].split(":")[0]})  # subtypes stripped (CoNLL-18)
    if cur:
        sents.append({"meta": meta, "words": cur})
    return sents


def write_conllu(path, gold, pred):
    with open(path, "w", encoding="utf-8") as f:
        for g, p in zip(gold, pred):
            for i, (gw, pw) in enumerate(zip(g["words"], p), 1):
                f.write(f"{i}\t{gw['form']}\t_\t{pw['upos']}\t_\t_\t{pw['head']}\t{pw['deprel']}\t_\t_\n")
            f.write("\n")


# ----------------------------------------------------------------- data
def fetch_treebanks(langs, tag, root):
    """Download only the test files over HTTPS (no git needed)."""
    import urllib.request
    root.mkdir(exist_ok=True)
    for lang in langs:
        for repo, fname in CONFIG[lang]["tests"].values():
            dest = root / repo / fname
            if dest.exists() and dest.stat().st_size > 0:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://raw.githubusercontent.com/UniversalDependencies/{repo}/{tag}/{fname}"
            print(f"  downloading {repo}/{fname}@{tag}")
            try:
                tmp = dest.with_suffix(".part")
                urllib.request.urlretrieve(url, tmp)
                tmp.replace(dest)
            except Exception as e:
                sys.exit(f"Could not download {url}\n  ({e})\n"
                         f"Download it in your browser and save it as {dest.resolve()}")


# ----------------------------------------------------------------- scoring
def sent_counts(gold, pred):
    """Per-sentence [tokens, upos_ok, uas_ok, las_ok]."""
    rows = []
    for g, p in zip(gold, pred):
        n = u = a = l = 0
        for gw, pw in zip(g["words"], p):
            n += 1
            u += gw["upos"] == pw["upos"]
            h = gw["head"] == pw["head"]
            a += h
            l += h and gw["deprel"] == pw["deprel"]
        rows.append((n, u, a, l))
    return rows


def totals(rows):
    n = sum(r[0] for r in rows)
    return {"tokens": n, "UPOS": 100 * sum(r[1] for r in rows) / n,
            "UAS": 100 * sum(r[2] for r in rows) / n, "LAS": 100 * sum(r[3] for r in rows) / n}


def per_relation(gold, pred):
    """LAS-style P/R/F1 per (base) relation: correct = right label AND right head."""
    tp, gc, pc = Counter(), Counter(), Counter()
    for g, p in zip(gold, pred):
        for gw, pw in zip(g["words"], p):
            gc[gw["deprel"]] += 1
            pc[pw["deprel"]] += 1
            if gw["deprel"] == pw["deprel"] and gw["head"] == pw["head"]:
                tp[gw["deprel"]] += 1
    out = {}
    for rel, n in gc.items():
        prec = tp[rel] / pc[rel] if pc[rel] else 0.0
        rec = tp[rel] / n
        out[rel] = {"gold": n, "P": 100 * prec, "R": 100 * rec,
                    "F1": 100 * (2 * prec * rec / (prec + rec)) if prec + rec else 0.0}
    return out


def by_bucket(gold, pred):
    """LAS by sentence length and by gold dependency distance."""
    sl, dd = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    sb = lambda n: "1-10" if n <= 10 else "11-20" if n <= 20 else "21-30" if n <= 30 else "31-40" if n <= 40 else "41+"
    db = lambda d: "root" if d == 0 else "1" if d == 1 else "2" if d == 2 else "3-6" if d <= 6 else "7+"
    for g, p in zip(gold, pred):
        b = sb(len(g["words"]))
        for i, (gw, pw) in enumerate(zip(g["words"], p), 1):
            ok = gw["head"] == pw["head"] and gw["deprel"] == pw["deprel"]
            sl[b][0] += 1; sl[b][1] += ok
            k = db(0 if gw["head"] == 0 else abs(gw["head"] - i))
            dd[k][0] += 1; dd[k][1] += ok
    f = lambda d: {k: {"n": v[0], "LAS": 100 * v[1] / v[0]} for k, v in d.items()}
    return {"sentence_length": f(sl), "dependency_distance": f(dd)}


def paired_bootstrap(rows_a_list, rows_b_list, metric_idx=3, B=10000, seed=12345):
    """
    Paired sentence-level bootstrap on LAS (metric_idx 3) or UAS (2).
    rows_*_list: list over test sets (each a list of per-sentence rows, same order in a and b).
    Resamples sentences within each test set; statistic = mean over sets of (A - B) score,
    i.e. the macro difference. Returns delta, 95% CI, two-sided p (H0: delta = 0).
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    obs, boots = [], []
    for ra, rb in zip(rows_a_list, rows_b_list):
        n = np.array([r[0] for r in ra], float)
        ca = np.array([r[metric_idx] for r in ra], float)
        cb = np.array([r[metric_idx] for r in rb], float)
        obs.append(100 * (ca.sum() - cb.sum()) / n.sum())
        idx = rng.integers(0, len(n), size=(B, len(n)))
        boots.append(100 * (ca[idx].sum(1) - cb[idx].sum(1)) / n[idx].sum(1))
    obs = float(np.mean(obs))
    boots = np.mean(np.stack(boots), axis=0)
    centred = boots - boots.mean()                     # null distribution by shifting
    p = float((np.abs(centred) >= abs(obs)).mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"delta": obs, "ci95": [float(lo), float(hi)], "p": max(p, 1 / B), "B": B}


# ----------------------------------------------------------------- stanza
def stanza_pipeline(lang, package, device, pretokenized):
    import stanza
    stanza.download(lang, package=package, processors=PROCESSORS, verbose=False)
    nlp = stanza.Pipeline(lang, package=package, processors=PROCESSORS,
                          tokenize_pretokenized=pretokenized, use_gpu=(device == "gpu"),
                          download_method=None, verbose=False)
    resolved = {}
    for name, proc in nlp.processors.items():
        try:
            resolved[name] = os.path.basename(proc.config.get("model_path", "") or "")
        except Exception:
            resolved[name] = "?"
    return nlp, resolved


def tag_parse(nlp, gold, batch=500):
    pred = []
    for i in range(0, len(gold), batch):
        chunk = [[w["form"] for w in s["words"]] for s in gold[i:i + batch]]
        doc = nlp(chunk)
        for s, g in zip(doc.sentences, gold[i:i + batch]):
            words = [{"upos": w.upos, "head": int(w.head), "deprel": (w.deprel or "_").split(":")[0]}
                     for w in s.words]
            assert len(words) == len(g["words"]), "token misalignment (MWT expansion?)"
            pred.append(words)
    assert len(pred) == len(gold)
    return pred


# ----------------------------------------------------------------- speed
def time_runs(fn, runs, warmup=1):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(runs):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    q = statistics.quantiles(ts, n=4) if len(ts) >= 4 else [min(ts), statistics.median(ts), max(ts)]
    return {"runs_s": ts, "median_s": statistics.median(ts), "best_s": min(ts),
            "iqr_s": [q[0], q[-1]]}


def speed(args, tb_root, device):
    gold = read_conllu(tb_root / "UD_English-EWT" / "en_ewt-ud-test.conllu")[: args.speed_sents]
    texts = []
    for s in gold:
        t = next((m.split("=", 1)[1].strip() for m in s["meta"] if m.startswith("# text")), None)
        texts.append(t or " ".join(w["form"] for w in s["words"]))
    raw = "\n\n".join(texts)
    n_tok = sum(len(s["words"]) for s in gold)
    out = {"sentences": len(gold), "gold_tokens": n_tok, "device": device, "runs": args.speed_runs}

    import spacy, torch
    if device == "cpu":
        out["torch_threads"] = torch.get_num_threads()
    sp = {}
    for name in ["en_core_web_sm"] + (["en_core_web_trf"] if args.spacy_trf else []):
        if device == "gpu":
            spacy.require_gpu()
        nlp_sp = spacy.load(name, exclude=["ner"])
        sp[name] = time_runs(lambda: nlp_sp(raw), args.speed_runs)
        sp[name]["tokens_per_s"] = n_tok / sp[name]["median_s"]
    out["spacy"] = sp

    st = {}
    for label, pkg in CONFIG["en"]["models"].items():
        nlp, resolved = stanza_pipeline("en", pkg, device, pretokenized=False)
        r = time_runs(lambda: nlp(raw), args.speed_runs)
        r["tokens_per_s"] = n_tok / r["median_s"]
        r["x_spacy_sm"] = r["median_s"] / sp["en_core_web_sm"]["median_s"]
        r["x_spacy_sm_best_of"] = r["best_s"] / sp["en_core_web_sm"]["best_s"]  # comparable to v2 poster
        if "en_core_web_trf" in sp:
            r["x_spacy_trf"] = r["median_s"] / sp["en_core_web_trf"]["median_s"]
        r["resolved_models"] = resolved
        st[label] = r
        # per-processor breakdown for the pooled model
        if label == CONFIG["en"]["pooled"]:
            prof = {}
            for proc in PROCESSORS.split(","):
                procs = PROCESSORS.split(",")[: PROCESSORS.split(",").index(proc) + 1]
                p2 = __import__("stanza").Pipeline("en", package=pkg, processors=",".join(procs),
                                                   use_gpu=(device == "gpu"), download_method=None, verbose=False)
                prof[proc] = time_runs(lambda: p2(raw), max(3, args.speed_runs // 2))["median_s"]
            prev, st[label]["cumulative_s_by_processor"] = 0.0, {}
            for proc, t in prof.items():
                st[label]["cumulative_s_by_processor"][proc] = t
            st[label]["share_by_processor"] = {}
            for proc, t in prof.items():
                st[label]["share_by_processor"][proc] = max(t - prev, 0) / list(prof.values())[-1]
                prev = t
    out["stanza"] = st
    return out


# ----------------------------------------------------------------- main
def env_info():
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "processor": platform.processor(), "cpu_count": os.cpu_count()}
    for mod in ["stanza", "spacy", "torch", "numpy"]:
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    try:
        import torch
        info["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


def accuracy(lang, args, tb_root, out_dir):
    cfg = CONFIG[lang]
    gold = {t: read_conllu(tb_root / repo / f) for t, (repo, f) in cfg["tests"].items()}
    res = {"tests": {t: {"sentences": len(g), "tokens": sum(len(s["words"]) for s in g)} for t, g in gold.items()},
           "models": {}, "trained_on": cfg["trained_on"]}
    rows = {}
    for label, pkg in cfg["models"].items():
        try:
            nlp, resolved = stanza_pipeline(lang, pkg, args.device, pretokenized=True)
        except Exception as e:
            print(f"  [skip] {lang}/{label} ({pkg}): {e}")
            res["models"][label] = {"error": str(e)}
            continue
        m = {"package": pkg, "resolved_models": resolved, "scores": {}, "relations": {}, "buckets": {}}
        rows[label] = {}
        for t, g in gold.items():
            print(f"  {lang} {label:>9} on {t}")
            pred = tag_parse(nlp, g)
            write_conllu(out_dir / f"{lang}_{label}_{t}.conllu", g, pred)
            r = sent_counts(g, pred)
            rows[label][t] = r
            m["scores"][t] = totals(r)
            m["relations"][t] = per_relation(g, pred)
            m["buckets"][t] = by_bucket(g, pred)
        heldout = [t for t in gold if t not in cfg["trained_on"][label]]
        m["macro"] = {k: statistics.mean(m["scores"][t][k] for t in gold) for k in ["UPOS", "UAS", "LAS"]}
        m["heldout_mean"] = {k: statistics.mean(m["scores"][t][k] for t in heldout) for k in ["UPOS", "UAS", "LAS"]}
        res["models"][label] = m

    # significance: pooled vs each single-treebank model
    pooled = cfg["pooled"]
    sig = {}
    if pooled in rows:
        common_heldout = [t for t in gold if all(t not in cfg["trained_on"][l] for l in rows)]
        for other in rows:
            if other == pooled:
                continue
            s = {}
            for t in gold:
                s[t] = {"LAS": paired_bootstrap([rows[pooled][t]], [rows[other][t]], 3, args.boot),
                        "UAS": paired_bootstrap([rows[pooled][t]], [rows[other][t]], 2, args.boot)}
            if common_heldout:
                s["heldout_mean"] = {"sets": common_heldout,
                                     "LAS": paired_bootstrap([rows[pooled][t] for t in common_heldout],
                                                             [rows[other][t] for t in common_heldout], 3, args.boot)}
            sig[f"{pooled}_vs_{other}"] = s
    res["significance"] = sig
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="en,de,fr")
    ap.add_argument("--ud-tag", default=UD_TAG_DEFAULT)
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--speed-sents", type=int, default=500)
    ap.add_argument("--speed-runs", type=int, default=5)
    ap.add_argument("--spacy-trf", action="store_true")
    ap.add_argument("--speed-only", action="store_true")
    ap.add_argument("--no-speed", action="store_true")
    ap.add_argument("--out", default="results_extended.json")
    args = ap.parse_args()

    langs = [l for l in args.langs.split(",") if l]
    tb_root, out_dir = Path("ud"), Path("preds")
    out_dir.mkdir(exist_ok=True)
    fetch_treebanks(sorted(set(langs) | {"en"}), args.ud_tag, tb_root)

    out_path = Path(args.out)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    results.setdefault("env", {})[args.device] = env_info()
    results["ud_tag"] = args.ud_tag
    results["protocol"] = {"tokenization": "gold (syntactic words)", "deprel": "subtypes stripped",
                           "bootstrap": "paired, sentence-level, two-sided", "processors": PROCESSORS}

    if not args.speed_only:
        results.setdefault("accuracy", {})
        for lang in langs:
            print(f"== accuracy: {lang}")
            results["accuracy"][lang] = accuracy(lang, args, tb_root, out_dir)
            out_path.write_text(json.dumps(results, indent=1))
    if not args.no_speed:
        print(f"== speed: {args.device}")
        results.setdefault("speed", {})[args.device] = speed(args, tb_root, args.device)
    out_path.write_text(json.dumps(results, indent=1))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
