# Stanza Revisited: Are the 2020 Limitations Resolved?

Research poster, NLP module, University of Trier (Usman Tariq).

Re-tests two limitations named in the Stanza paper (Qi et al., 2020) with Stanza 1.14.0:

- **L1 (single-dataset models):** does Stanza's pooled ("combined") default model parse *unseen*
  UD treebanks better than single-treebank models, without losing accuracy on its training domains?
  English (ewt, gum, combined), French (gsd, sequoia, combined), German (gsd, hdt, combined).
- **L2 (speed):** how large is the CPU runtime gap to spaCy (`en_core_web_sm`), and which stage causes it?

## Files

| File | What it is |
|---|---|
| `run_extended.py` | Experiment script: accuracy, bootstrap significance tests, error analysis, speed |
| `results_extended.json` | Results behind every number on the poster, incl. software versions and hardware |

## Reproduce

```
pip install stanza spacy numpy
python -m spacy download en_core_web_sm
python run_extended.py                 # accuracy (en, de, fr) + CPU speed
```

The script downloads the UD 2.15 test files and the Stanza models itself. Results are written to
`results_extended.json` (all scores, bootstrap tests, per-relation and distance breakdowns, timings,
software versions and hardware) and predictions to `preds/` as CoNLL-U.

## Protocol

- Gold tokenization (syntactic words); UPOS, UAS, LAS with relation subtypes stripped (CoNLL 2018).
- Paired sentence-level bootstrap, 10,000 resamples, two-sided; 95% interval of the LAS difference.
- Speed: first 500 EWT test sentences as raw text, full Stanza pipeline (tokenize, pos, lemma, depparse)
  vs spaCy `en_core_web_sm` without NER; median of 5 runs after one warm-up run.

## Note on the "combined" models

Training data of Stanza's combined models (from `stanza/utils/datasets/prepare_tokenizer_treebank.py`, tag v1.14.0):
English: EWT, GUM, GUMReddit (train) + PUD, Pronouns; French: GSD, Sequoia, ParisStories, Rhapsodie;
German: **GSD only**, so the German comparison serves as a same-data control rather than a pooling test.
