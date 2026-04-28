import spacy
import pandas as pd
from ner_pipeline import evaluate_ner 

def setup_custom_ner(position="before"):
    nlp = spacy.load("en_core_web_sm")
    if "entity_ruler" in nlp.pipe_names:
        nlp.remove_pipe("entity_ruler")

    ruler = nlp.add_pipe("entity_ruler", before="ner" if position == "before" else None, after="ner" if position == "after" else None)

    patterns = [
        {"label": "CLIMATE_EVENT", "pattern": "COP28"},
        {"label": "CLIMATE_EVENT", "pattern": "COP27"},
        {"label": "CLIMATE_EVENT", "pattern": "COP26"},
        {"label": "POLICY", "pattern": "Paris Agreement"},
        {"label": "POLICY", "pattern": "Kyoto Protocol"},
        {"label": "POLICY", "pattern": [{"TEXT": "European"}, {"TEXT": "Green"}, {"TEXT": "Deal"}]},
        {"label": "REPORT", "pattern": "IPCC AR6"},
        {"label": "REPORT", "pattern": "Sixth Assessment Report"},
        {"label": "THRESHOLD", "pattern": "1.5°C"},
        {"label": "THRESHOLD", "pattern": [{"IS_DIGIT": True}, {"TEXT": "degrees"}, {"TEXT": "Celsius"}]}
    ]
    ruler.add_patterns(patterns)
    return nlp

def get_entities_dataframe(nlp_model, texts_df):
    all_ents = []
    for _, row in texts_df.iterrows():
        t_id = row['id'] 
        doc = nlp_model(row['text'])
        for ent in doc.ents:
            all_ents.append({"text_id": t_id, "entity_text": ent.text, "entity_label": ent.label_})
    return pd.DataFrame(all_ents)

def evaluate_delta():
    gold_df = pd.read_csv("data/gold_entities.csv")
    df = pd.read_csv("data/climate_articles.csv")
    english_texts = df[df['language'] == 'en']

    nlp_base = spacy.load("en_core_web_sm")
    nlp_custom = setup_custom_ner(position="before")

    print("\n" + "="*50)
    print("LIVE DEMO: CUSTOM RULES IN ACTION")
    print("="*50)
    
    sample_text = english_texts.iloc[0]['text'] 
    doc_base = nlp_base(sample_text)
    doc_custom = nlp_custom(sample_text)

    print(f"Text Sample: {sample_text[:100]}...")
    print("\nComparison:")
    print(f"{'Entity':<25} | {'Base Label':<15} | {'Custom Label':<15}")
    print("-" * 60)
    
    for ent_custom in doc_custom.ents:
        base_label = next((e.label_ for e in doc_base.ents if e.text == ent_custom.text), "NOT FOUND")
        if ent_custom.label_ != base_label: 
            print(f"{ent_custom.text:<25} | {base_label:<15} | {ent_custom.label_:<15} << CHANGED!")
        else:
            print(f"{ent_custom.text:<25} | {base_label:<15} | {ent_custom.label_:<15}")

    print("\n" + "="*50)
    print("EVALUATING ENTIRE DATASET...")
    base_ents = get_entities_dataframe(nlp_base, english_texts)
    metrics_base = evaluate_ner(base_ents, gold_df)

    custom_ents = get_entities_dataframe(nlp_custom, english_texts)
    metrics_custom = evaluate_ner(custom_ents, gold_df)

    print("\nSTRETCH TASK: EVALUATION DELTA")
    print(f"Base Model Metrics:   {metrics_base}")
    print(f"Custom Model Metrics: {metrics_custom}")
    print("="*50)

if __name__ == "__main__":
    evaluate_delta()