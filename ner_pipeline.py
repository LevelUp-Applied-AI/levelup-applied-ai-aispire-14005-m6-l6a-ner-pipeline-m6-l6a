"""
Module 6 Week A — Lab: NER Pipeline
"""

import pandas as pd
import numpy as np
import spacy
import unicodedata
from transformers import pipeline as hf_pipeline



def load_data(filepath="data/climate_articles.csv"):
    df = pd.read_csv(filepath)
    return df


def explore_data(df):
    shape = df.shape

    lang_counts = df["language"].value_counts().to_dict()
    category_counts = df["category"].value_counts().to_dict()

    text_lengths = df["text"].dropna().apply(lambda x: len(x.split()))

    text_length_stats = {
        "mean": text_lengths.mean(),
        "min": text_lengths.min(),
        "max": text_lengths.max()
    }

    return {
        "shape": shape,
        "lang_counts": lang_counts,
        "category_counts": category_counts,
        "text_length_stats": text_length_stats
    }



def preprocess_text(text, nlp):
    text = unicodedata.normalize("NFC", text)
    doc = nlp(text)

    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_punct and not token.is_space
    ]

    return tokens



def extract_spacy_entities(df, nlp):
    results = []
    df_en = df[df["language"] == "en"]

    for _, row in df_en.iterrows():
        doc = nlp(row["text"])

        for ent in doc.ents:
            results.append({
                "text_id": row["id"],
                "entity_text": ent.text,
                "entity_label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char
            })

    return pd.DataFrame(results)



def extract_hf_entities(df, ner_pipeline):
    results = []
    df_en = df[df["language"] == "en"]

    for _, row in df_en.iterrows():
        text_id = row["id"]
        text = row["text"]

        ner_results = ner_pipeline(text)

        current_word = ""
        current_label = ""
        start_char = None
        end_char = None

        for ent in ner_results:
            word = ent["word"]
            label = ent["entity"]
            start = ent["start"]
            end = ent["end"]

            if word.startswith("##"):
                current_word += word[2:]
                end_char = end
            else:
                if current_word:
                    results.append({
                        "text_id": text_id,
                        "entity_text": current_word,
                        "entity_label": current_label.replace("B-", "").replace("I-", ""),
                        "start_char": start_char,
                        "end_char": end_char
                    })

                current_word = word
                current_label = label
                start_char = start
                end_char = end

        if current_word:
            results.append({
                "text_id": text_id,
                "entity_text": current_word,
                "entity_label": current_label.replace("B-", "").replace("I-", ""),
                "start_char": start_char,
                "end_char": end_char
            })

    return pd.DataFrame(results)



def compare_ner_outputs(spacy_df, hf_df):
    spacy_counts = spacy_df["entity_label"].value_counts().to_dict()
    hf_counts = hf_df["entity_label"].value_counts().to_dict()

    total_spacy = len(spacy_df)
    total_hf = len(hf_df)

    spacy_set = set(zip(spacy_df["text_id"], spacy_df["entity_text"]))
    hf_set = set(zip(hf_df["text_id"], hf_df["entity_text"]))

    both = spacy_set & hf_set
    spacy_only = spacy_set - hf_set
    hf_only = hf_set - spacy_set

    return {
        "spacy_counts": spacy_counts,
        "hf_counts": hf_counts,
        "total_spacy": total_spacy,
        "total_hf": total_hf,
        "both": both,
        "spacy_only": spacy_only,
        "hf_only": hf_only
    }



def evaluate_ner(predicted_df, gold_df):
    pred_set = set(zip(
        predicted_df["text_id"],
        predicted_df["entity_text"],
        predicted_df["entity_label"]
    ))

    gold_set = set(zip(
        gold_df["text_id"],
        gold_df["entity_text"],
        gold_df["entity_label"]
    ))

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


if __name__ == "__main__":
    # Load spaCy and HF models once, reuse across functions
    nlp = spacy.load("en_core_web_sm")
    hf_ner = hf_pipeline("ner", model="dslim/bert-base-NER")

    # Load and explore
    df = load_data()
    if df is not None:
        summary = explore_data(df)
        if summary is not None:
            print(f"Shape: {summary['shape']}")
            print(f"Languages: {summary['lang_counts']}")
            print(f"Categories: {summary['category_counts']}")
            print(f"Text length (words): {summary['text_length_stats']}")

        # Preprocess a sample to verify your function
        sample_row = df[df["language"] == "en"].iloc[0]
        sample_tokens = preprocess_text(sample_row["text"], nlp)
        if sample_tokens is not None:
            print(f"\nSample preprocessed tokens: {sample_tokens[:10]}")

        # spaCy NER across the English corpus
        spacy_entities = extract_spacy_entities(df, nlp)
        if spacy_entities is not None:
            print(f"\nspaCy entities: {len(spacy_entities)} total")

        # HF NER across the English corpus
        hf_entities = extract_hf_entities(df, hf_ner)
        if hf_entities is not None:
            print(f"HF entities: {len(hf_entities)} total")

        # Compare the two systems
        if spacy_entities is not None and hf_entities is not None:
            comparison = compare_ner_outputs(spacy_entities, hf_entities)
            if comparison is not None:
                print(f"\nBoth systems agreed on {len(comparison['both'])} entities")
                print(f"spaCy-only: {len(comparison['spacy_only'])}")
                print(f"HF-only: {len(comparison['hf_only'])}")

        # Evaluate against gold standard
        gold = pd.read_csv("data/gold_entities.csv")
        if spacy_entities is not None:
            metrics = evaluate_ner(spacy_entities, gold)
            if metrics is not None:
                print(f"\nspaCy evaluation: {metrics}")
