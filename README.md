# Lab 6A — NER Pipeline

Module 6 Week A lab for AI.SPIRE Applied AI & ML Systems.

Build and compare Named Entity Recognition (NER) pipelines using spaCy and Hugging Face on climate-related text data.

## Setup

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download xx_ent_wiki_sm
```

The Hugging Face NER pipeline runs on PyTorch; we install the CPU wheel explicitly so the download stays small. `requirements.txt` intentionally omits `torch`. The spaCy English model is ~12 MB; the multilingual model (`xx_ent_wiki_sm`) is ~11 MB. The first run of the HF NER model will download ~250 MB of model weights — this is a one-time download.

## Tasks

Complete the eight functions in `ner_pipeline.py`:
1. `load_data(filepath)` — Load the climate articles dataset
2. `explore_data(df)` — Return a summary dict (shape, language/category counts, text length stats)
3. `preprocess_text(text, nlp)` — NFC-normalize and return lowercased lemmas using the injected spaCy pipeline
4. `extract_spacy_entities(df, nlp)` — Extract entities using spaCy NER
5. `extract_hf_entities(df, ner_pipeline)` — Extract entities using Hugging Face NER (merge `##` subwords and strip `B-`/`I-` IOB prefix)
6. `compare_ner_outputs(spacy_df, hf_df)` — Entity counts per system plus `both`/`spacy_only`/`hf_only` overlap sets
7. `evaluate_ner(predicted_df, gold_df)` — Compute entity-level precision, recall, F1 (filter `predicted_df` to gold text_ids first)
8. `extract_multilingual_entities(df, multilingual_nlp)` — Extract entities from the Arabic subset of the corpus using the multilingual spaCy model (`xx_ent_wiki_sm`)

## Submission

1. Create a branch: `lab-6a-ner-pipeline`
2. Complete `ner_pipeline.py`
3. Open a PR to `main`
4. Paste your PR URL into TalentLMS → Module 6 Week A → Lab 6A

---

## License

This repository is provided for educational use only. See [LICENSE](LICENSE) for terms.

You may clone and modify this repository for personal learning and practice, and reference code you wrote here in your professional portfolio. Redistribution outside this course is not permitted.
