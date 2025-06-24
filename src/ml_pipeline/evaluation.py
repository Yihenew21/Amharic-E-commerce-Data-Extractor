import os
import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
import numpy as np

# --- Configuration ---
MODELS_TO_EVALUATE = [
    "models/bert-base-multilingual-cased_finetuned",
    "models/distilbert-base-multilingual-cased_finetuned",
    "models/xlm-roberta-base_finetuned"
]
DATA_DIR = "data/labeled/v1"
RESULTS_OUTPUT_PATH = "models/model_cards/evaluation_results.csv"

# Replicate the label generation from the training script to ensure consistency
BASE_ENTITY_TYPES = ["PRODUCT", "PRICE", "LOCATION", "BRAND", "SIZE", "CONTACT"]
LABEL_NAMES = ["O"] + [f"B-{t}" for t in BASE_ENTITY_TYPES] + [f"I-{t}" for t in BASE_ENTITY_TYPES]
id_to_label = {i: label for i, label in enumerate(LABEL_NAMES)}
label_to_id = {label: i for i, label in enumerate(LABEL_NAMES)}

def load_conll_data_for_eval(file_path):
    """Loads a single CoNLL file into a Hugging Face Dataset."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: {file_path} not found.")

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    all_tokens, all_ner_tags = [], []
    current_tokens, current_ner_tags = [], []
    
    for line in lines:
        line = line.strip()
        if line:
            parts = line.split('\t')
            if len(parts) == 2:
                current_tokens.append(parts[0])
                current_ner_tags.append(parts[1])
        else:
            if current_tokens:
                all_tokens.append(current_tokens)
                all_ner_tags.append(current_ner_tags)
                current_tokens, current_ner_tags = [], []
    
    if current_tokens:
        all_tokens.append(current_tokens)
        all_ner_tags.append(current_ner_tags)

    return Dataset.from_dict({"tokens": all_tokens, "ner_tags": all_ner_tags})

def main():
    """
    Main function to evaluate all specified models by performing manual, aligned inference.
    """
    print("Starting model evaluation...")
    test_file_path = os.path.join(DATA_DIR, "test.conll")
    print(f"Loading test data from: {test_file_path}")
    test_dataset = load_conll_data_for_eval(test_file_path)

    evaluation_results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    for model_path in MODELS_TO_EVALUATE:
        model_name = os.path.basename(model_path)
        print(f"\n--- Evaluating Model: {model_name} ---")

        if not os.path.exists(model_path):
            print(f"Warning: Model path not found: {model_path}. Skipping.")
            continue
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForTokenClassification.from_pretrained(model_path).to(device)

            def tokenize_and_align_labels(examples):
                tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
                labels = []
                for i, tags in enumerate(examples["ner_tags"]):
                    word_ids = tokenized_inputs.word_ids(batch_index=i)
                    previous_word_idx = None
                    label_ids = []
                    for word_idx in word_ids:
                        if word_idx is None or word_idx == previous_word_idx:
                            label_ids.append(-100)
                        else:
                            label_ids.append(label_to_id[tags[word_idx]])
                        previous_word_idx = word_idx
                    labels.append(label_ids)
                tokenized_inputs["labels"] = labels
                return tokenized_inputs

            print("Tokenizing and aligning test data...")
            tokenized_test_dataset = test_dataset.map(tokenize_and_align_labels, batched=True)
            tokenized_test_dataset = tokenized_test_dataset.remove_columns(['tokens', 'ner_tags'])

            data_collator = DataCollatorForTokenClassification(tokenizer)
            test_dataloader = DataLoader(tokenized_test_dataset, batch_size=8, collate_fn=data_collator)

            model.eval()
            all_predictions, all_true_labels = [], []

            print(f"Running inference with {model_name}...")
            for batch in test_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = model(**batch)
                
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=2)
                
                predictions = predictions.cpu().numpy()
                labels = batch["labels"].cpu().numpy()

                true_labels_batch = [[id_to_label[l] for l in label if l != -100] for label in labels]
                predictions_batch = [
                    [id_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
                    for prediction, label in zip(predictions, labels)
                ]
                
                all_predictions.extend(predictions_batch)
                all_true_labels.extend(true_labels_batch)
            
            precision = precision_score(all_true_labels, all_predictions)
            recall = recall_score(all_true_labels, all_predictions)
            f1 = f1_score(all_true_labels, all_predictions)
            
            print(f"\nResults for {model_name}:")
            print(f"  Precision: {precision:.4f}\n  Recall: {recall:.4f}\n  F1-Score: {f1:.4f}")
            print("\nClassification Report:")
            print(classification_report(all_true_labels, all_predictions, digits=4))

            evaluation_results.append({
                "model_name": model_name, "f1_score": f1, "precision": precision, "recall": recall
            })

        except Exception as e:
            print(f"An error occurred while evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()

    if evaluation_results:
        print("\n--- Overall Model Comparison ---")
        results_df = pd.DataFrame(evaluation_results).sort_values(by="f1_score", ascending=False)
        print(results_df.to_string(index=False))

        os.makedirs(os.path.dirname(RESULTS_OUTPUT_PATH), exist_ok=True)
        results_df.to_csv(RESULTS_OUTPUT_PATH, index=False)
        print(f"\nEvaluation results saved to {RESULTS_OUTPUT_PATH}")
    else:
        print("\nNo models were evaluated. Check model paths and data files.")

if __name__ == "__main__":
    main()
