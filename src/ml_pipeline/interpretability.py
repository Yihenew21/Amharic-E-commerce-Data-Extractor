import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from lime.lime_text import LimeTextExplainer

# Define the model and tokenizer
MODEL_PATH = "models/distilbert-base-multilingual-cased_finetuned"
ENTITY_TYPE = "PRODUCT"

# Entity types (update as needed)
ENTITY_TYPES = [
    "PRODUCT", "PRICE", "LOCATION", "BRAND", "SIZE", "CONTACT"
]

# Function to return 1 if the entity is present in the sentence, else 0
def predict_entity_presence(texts, entity_type):
    nlp = pipeline("token-classification", model=MODEL_PATH, tokenizer=MODEL_PATH, aggregation_strategy="simple")
    results = []
    for text in texts:
        entities = nlp(text)
        found = any(ent['entity_group'] == entity_type for ent in entities)
        results.append([1 if found else 0])
    return np.array(results)

def explain_with_lime(text, entity_type):
    print(f"\n--- LIME Explanation for Entity: {entity_type} ---")
    explainer = LimeTextExplainer(split_expression=" ")
    def predict_fn(texts):
        return predict_entity_presence(texts, entity_type)
    exp = explainer.explain_instance(text, predict_fn, num_features=len(text.split()), labels=[0])
    print(f"Text: {text}")
    print(f"Word importances for predicting presence of entity '{entity_type}':")
    for word, score in exp.as_list(label=0):
        print(f"  {word}: {score:.3f}")


def main():
    # Sample Amharic sentences
    sample_texts = [
        "አዲስ አበባ ውስጥ የሚሸጥ Lg tv ዋጋ 15,000 ብር ነው",
        "የህፃን ዳይፐር በካርቶን 2500 ብር",
        "ቦሌ አካባቢ የሚከራይ ቤት",
        "ለበለጠ መረጃ በ 0912345678 ይደውሉ"
    ]
    for text in sample_texts:
        explain_with_lime(text, ENTITY_TYPE)

if __name__ == "__main__":
    main()
