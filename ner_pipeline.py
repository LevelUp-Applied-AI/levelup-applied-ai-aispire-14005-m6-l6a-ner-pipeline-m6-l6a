"""
Module 6 Week A — Lab: NER Pipeline

Build and compare Named Entity Recognition pipelines using spaCy
and Hugging Face on climate-related text data.

Run: python ner_pipeline.py
"""

import unicodedata

import pandas as pd
import numpy as np
import spacy
from transformers import pipeline as hf_pipeline


def load_data(filepath="data/climate_articles.csv"):
    """Load the climate articles dataset."""
    df = pd.read_csv(filepath)
    return df
def explore_data(df):
    """
    Summarize basic corpus statistics for the dataset.
    """
    shape = df.shape
    
    lang_counts = df['language'].value_counts().to_dict()
    
    category_counts = df['category'].value_counts().to_dict()
    
    word_counts = df['text'].str.split().str.len()
    
    text_length_stats = {
        'mean': word_counts.mean(),
        'min': word_counts.min(),
        'max': word_counts.max()
    }
    
    return {
        'shape': shape,
        'lang_counts': lang_counts,
        'category_counts': category_counts,
        'text_length_stats': text_length_stats
    }


def preprocess_text(text, nlp):
    """
    Clean text: normalize Unicode, remove punctuation/spaces, and lemmatize.
    """
    print("Task 2: Preprocess a sample text with spaCy  ")
    # 1. Apply Unicode normalization (NFC)
    # This ensures characters like 'e' with an accent are handled consistently
    # NFC (Normalization Form C) composes characters into a single code point when possible
    # like ( café ) becomes a single 'é' character instead of 'e' + combining accent character
    normalized_text = unicodedata.normalize("NFC", text)
    print(f"Original text:   {text[:100]}... | Original length: {len(text)}")
    print(f"Normalized text: {normalized_text[:100]}... | Normalized length: {len(normalized_text)}")
    print("-"*60)
    
    # 2. Process the normalized text through the spaCy pipeline
    doc = nlp(normalized_text)
    print(f"{'Token':<15} | {'Lemma':<15} | {'Punct?':<8} | {'Space?':<8}")
    print("-" * 52)
    for token in doc[:10]:
        print(f"{token.text:<15} | {token.lemma_:<15} | {str(token.is_punct):<8} | {str(token.is_space):<8}")
    
    print("-" * 60)
 
    
    # 3. Create a list of lowercased lemmas
    # Filter: Keep tokens only if they are NOT punctuation and NOT whitespace
    clean_tokens = [
        token.lemma_.lower() 
        for token in doc 
        if not token.is_punct and not token.is_space
    ]
    
    return clean_tokens




def extract_spacy_entities(df, nlp):
    """"Extract named entities from English texts using spaCy NER."""

    print("Task 3: Extract entities with spaCy NER across the English corpus")
    english_df = df[df["language"] == "en"]
    
    entities_list = []
    
    for index, row in english_df.iterrows():
        text = row["text"]
        text_id = row["id"]
        
        doc = nlp(text)
        
        for ent in doc.ents:
            entities_list.append({
                "text_id": text_id,
                "entity_text": ent.text,
                "entity_label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char
            })
        
            
    return pd.DataFrame(entities_list)

def extract_hf_entities(df, ner_pipeline):
    english_df = df[df["language"] == "en"]
    entities_list = []
    print_count = 0 

    for index, row in english_df.iterrows():
        text = row["text"]
        text_id = row["id"]
        
        # Step 1: Run the model
        raw_results = ner_pipeline(text)
        
        # --- NEW: Print the raw tokens list to see how BERT "sees" the text ---
        if print_count < 4:  
            all_tokens = [token["word"] for token in raw_results]
            print(f"\n>>> Raw Tokens for Text ID {text_id}:")
            print(all_tokens)
            print("-" * 50)
        # ---------------------------------------------------------------------

        merged_entities = []
        current_entity = None

        for token in raw_results:
            word = token["word"]
            label = token["entity"]
            
            # Case 1: START of a new entity
            if label.startswith("B-"):
                if current_entity:
                    merged_entities.append(current_entity)
                
                current_entity = {
                    "text_id": text_id,
                    "entity_text": word,
                    "entity_label": label.replace("B-", ""),
                    "start_char": token["start"],
                    "end_char": token["end"]
                }
            
            # Case 2: CONTINUATION of the same entity
            elif label.startswith("I-") and current_entity:
                if word.startswith("##"):
                    current_entity["entity_text"] += word.replace("##", "")
                else:
                    current_entity["entity_text"] += " " + word
                current_entity["end_char"] = token["end"]
            
            # Case 3: OUTSIDE any entity
            else:
                if current_entity:
                    merged_entities.append(current_entity)
                    current_entity = None

        if current_entity:
            merged_entities.append(current_entity)
            
        entities_list.extend(merged_entities)
        print_count += 1

    return pd.DataFrame(entities_list)

    

def compare_ner_outputs(spacy_df, hf_df):
    # 1. Basic counts
    total_spacy = len(spacy_df)
    total_hf = len(hf_df)
    spacy_counts = spacy_df['entity_label'].value_counts().to_dict()
    hf_counts = hf_df['entity_label'].value_counts().to_dict()

    # 2. Find the overlap (both)
    # We merge to find common entities
    common = pd.merge(spacy_df, hf_df, on=['text_id', 'entity_text'], how='inner')
    
    # CRITICAL: The test expects a SET of (text_id, entity_text) tuples
    both = set(zip(common['text_id'], common['entity_text']))

    # 3. Find unique entities (Only spaCy)
    # We take all spaCy entities and subtract the ones found in both
    all_spacy = set(zip(spacy_df['text_id'], spacy_df['entity_text']))
    spacy_only = all_spacy - both

    # 4. Find unique entities (Only HF)
    all_hf = set(zip(hf_df['text_id'], hf_df['entity_text']))
    hf_only = all_hf - both

    # Return the dictionary exactly as requested by Task 5 and the tests
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
    # 1. Matching entities (True Positives)
    # We match on text_id, entity_text, AND entity_label
    matches = pd.merge(predicted_df, gold_df, on=['text_id', 'entity_text', 'entity_label'], how='inner')
    
    tp = len(matches) # True Positives
    fp = len(predicted_df) - tp # False Positives (Model guessed but wrong)
    fn = len(gold_df) - tp # False Negatives (Model missed it)

    # 2. Calculate Metrics
    # We use (tp + 1e-9) to avoid division by zero
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }    

if __name__ == "__main__":
    # Load spaCy and HF models once, reuse across functions
    nlp = spacy.load("en_core_web_sm")
    hf_ner = hf_pipeline("ner", model="dslim/bert-base-NER")
    #Task 1: Load and explore the dataset, then preprocess a sample text
    # Load and explore
    df = load_data()
    if df is not None:
        summary = explore_data(df)
        print ("="*120 + "\n" + "="*120)
        if summary is not None:
            print("Task 1: Dataset Summary")
            print(f"Shape: {summary['shape']}")
            print(f"Languages: {summary['lang_counts']}")
            print(f"Categories: {summary['category_counts']}")
            print(f"Text length (words): {summary['text_length_stats']}")
            print("=="*60)

    # Task 2: Implement and run the NER pipelines, then compare results
        # Preprocess a sample to verify your function
        sample_row = df[df["language"] == "en"].iloc[0]
        sample_tokens = preprocess_text(sample_row["text"], nlp)
        if sample_tokens is not None:
            print(f"\nSample preprocessed tokens: {sample_tokens[:10]}")
            print("=="*60)


        # task 3: Implement the NER extraction functions, run them, and compare outputs
        # spaCy NER across the English corpus
        spacy_entities = extract_spacy_entities(df, nlp)
        if spacy_entities is not None:
            print(f"\nspaCy entities: {len(spacy_entities)} total")
        # task 4: Implement the evaluation function and evaluate against gold standard
        # HF NER across the English corpus
        hf_entities = extract_hf_entities(df, hf_ner)
        if hf_entities is not None:
            print(f"HF entities: {len(hf_entities)} total")
            
        #task 5: Compare the two systems and evaluate against gold standard
        # Compare the two systems
        if spacy_entities is not None and hf_entities is not None:
            comparison = compare_ner_outputs(spacy_entities, hf_entities)
            if comparison is not None:
                print(f"\nBoth systems agreed on {len(comparison['both'])} entities")
                print(f"spaCy-only: {len(comparison['spacy_only'])}")
                print(f"HF-only: {len(comparison['hf_only'])}")
        # task 6: Implement the evaluation function and evaluate against gold standard    
        # Evaluate against gold standard
        gold = pd.read_csv("data/gold_entities.csv")
        if spacy_entities is not None:
            metrics = evaluate_ner(spacy_entities, gold)
            if metrics is not None:
                print(f"\nspaCy evaluation: {metrics}")
