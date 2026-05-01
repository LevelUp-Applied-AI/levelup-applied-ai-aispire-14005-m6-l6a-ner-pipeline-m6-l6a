import pandas as pd
import spacy
from collections import Counter
from transformers import pipeline

# -------------------------
# Load dataset
# -------------------------
df = pd.read_csv("data/climate_articles.csv")

# Assume column name is 'text'
texts = df['text'].dropna()

# -------------------------
# Simple language split
# -------------------------
def is_arabic(text):
    return any('\u0600' <= c <= '\u06FF' for c in text)

english_texts = [t for t in texts if not is_arabic(t)][:20]
arabic_texts = [t for t in texts if is_arabic(t)][:20]

# -------------------------
# Load models
# -------------------------
print("Loading spaCy model...")
try:
    nlp = spacy.load("xx_ent_wiki_sm")
except OSError:
    print("spaCy model 'xx_ent_wiki_sm' not found. Downloading it now...")
    from spacy.cli import download as spacy_download

    spacy_download("xx_ent_wiki_sm")
    nlp = spacy.load("xx_ent_wiki_sm")

print("Loading Hugging Face model...")
hf_ner = pipeline("ner", model="Davlan/xlm-roberta-base-wikiann-ner", aggregation_strategy="simple")

# -------------------------
# Helper function
# -------------------------
def process_spacy(texts):
    all_entities = []
    for doc in nlp.pipe(texts):
        for ent in doc.ents:
            all_entities.append((ent.text, ent.label_))
    return all_entities

def process_hf(texts):
    all_entities = []
    for text in texts:
        results = hf_ner(text)
        for ent in results:
            all_entities.append((ent['word'], ent['entity_group']))
    return all_entities

# -------------------------
# Run models
# -------------------------
print("Processing English...")
spacy_en = process_spacy(english_texts)
hf_en = process_hf(english_texts)

print("Processing Arabic...")
spacy_ar = process_spacy(arabic_texts)
hf_ar = process_hf(arabic_texts)

# -------------------------
# Analysis function
# -------------------------
def analyze(entities):
    counter = Counter([label for _, label in entities])
    total = len(entities)
    examples = entities[:3]
    return total, counter, examples


def entity_density(total_entities, texts):
    total_words = sum(len(t.split()) for t in texts)
    return (total_entities / total_words) * 100 if total_words > 0 else 0
# -------------------------
# Results
# -------------------------
results = {
    "spaCy English": analyze(spacy_en),
    "HF English": analyze(hf_en),
    "spaCy Arabic": analyze(spacy_ar),
    "HF Arabic": analyze(hf_ar),
}

# -------------------------
# Print results
# -------------------------
for name, (total, counter, examples) in results.items():
    print("\n", "="*40)
    print(name)
    print("Total entities:", total)
    print("Counts:", counter)
    print("Examples:", examples)