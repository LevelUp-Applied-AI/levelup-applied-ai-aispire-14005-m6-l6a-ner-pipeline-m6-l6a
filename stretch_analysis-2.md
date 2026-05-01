# Multilingual NER Comparison Analysis

## Labeling Approach

In this analysis, we use **Option A (native labels)**, meaning each model’s original entity labels are preserved without applying any mapping. This allows for a clearer comparison of how each model performs according to its own labeling schema.

---

## Comparison Summary

| Language | Model                | Total Entities | ORG | PER | LOC | MISC | Example Entities                                              |
| -------- | -------------------- | -------------- | --- | --- | --- | ---- | ------------------------------------------------------------- |
| English  | spaCy                | 93             | 39  | 12  | 27  | 15   | (IPCC, MISC), (Celsius, PER), (Report, MISC)                  |
| English  | Hugging Face (XLM-R) | 93             | 55  | 10  | 28  | -    | (Antonio Guterres, PER), (COP, ORG)                           |
| Arabic   | spaCy                | 20             | 2   | 9   | 2   | 7    | (وأكد التقرير, PER), (وقّع الأردن, MISC) |
| Arabic   | Hugging Face (XLM-R) | 64             | 33  | 2   | 29  | -    | (الأردن, LOC), (البنك الدولي, ORG)           |

---

## Analysis

Named Entity Recognition (NER) performance shows a clear difference between English and Arabic texts in the dataset. In English, both models perform relatively well, identifying entities such as "Antonio Guterres" (PER) and "COP" (ORG) correctly. The total number of detected entities is equal for both models (93), although the distribution differs slightly, with the Hugging Face model identifying more organizations. This indicates that both models are generally reliable for English NER tasks, even when using multilingual configurations.

However, the performance drops significantly when processing Arabic text, especially for the spaCy model. spaCy detects only 20 entities and frequently misclassifies non-entity phrases as named entities. For example, phrases like "وأكد التقرير" and "وقّع الأردن" are incorrectly labeled as entities, even though they are not proper names. This reflects the challenges of Arabic NLP, including the lack of capitalization, complex word structure, and limited representation in training data. In contrast, the Hugging Face model performs much better on Arabic, detecting 64 entities and correctly identifying meaningful ones such as "الأردن" (LOC) and "البنك الدولي" (ORG). Despite this improvement, it still struggles with certain entity types, such as person names, which appear less frequently.

These findings have important implications for NLP applications in the MENA region. In bilingual environments like Jordan, systems must process both English and Arabic text accurately. Using models that perform poorly on Arabic can result in missing or incorrect information extraction. The results suggest that transformer-based multilingual models like XLM-RoBERTa are more suitable for handling Arabic text compared to traditional multilingual pipelines like spaCy. However, to achieve production-level performance, additional improvements such as fine-tuning on Arabic datasets or integrating rule-based methods may be necessary.

---

## Key Takeaways

- Multilingual transformer models outperform traditional pipelines in Arabic NER.
- Arabic text presents unique challenges such as lack of capitalization and complex morphology.
- spaCy’s multilingual model struggles significantly with Arabic entity detection.
- Hugging Face models provide better cross-lingual generalization but still require improvement.
- Bilingual NLP systems in the MENA region must prioritize strong Arabic language support.
