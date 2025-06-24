import os
import pandas as pd
from datasets import DatasetDict, Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

# --- Configuration ---
MODEL_CHECKPOINT = "bert-base-multilingual-cased" # A good multilingual base model
DATA_DIR = "data/labeled/v1"
OUTPUT_DIR = "models/ner_model"
LOGGING_DIR = "logs"

# Define your base entity types used in annotation
BASE_ENTITY_TYPES = ["PRODUCT", "PRICE", "LOCATION", "BRAND", "SIZE", "CONTACT"]

# Automatically generate BIO labels from base entity types
LABEL_NAMES = ["O"] # "O" for Outside
for entity_type in BASE_ENTITY_TYPES:
    LABEL_NAMES.append(f"B-{entity_type}") # Beginning of an entity
    LABEL_NAMES.append(f"I-{entity_type}") # Inside an entity

# Map labels to IDs and vice-versa
label_to_id = {label: i for i, label in enumerate(LABEL_NAMES)}
id_to_label = {i: label for i, label in enumerate(LABEL_NAMES)}

print(f"Defined Labels ({len(LABEL_NAMES)}): {LABEL_NAMES}")
print(f"Label to ID Mapping: {label_to_id}")

def load_conll_data(data_dir):
    """Loads CoNLL files into a DatasetDict."""
    data_files = {
        "train": os.path.join(data_dir, "train.conll"),
        "validation": os.path.join(data_dir, "val.conll"),
        "test": os.path.join(data_dir, "test.conll"),
    }
    
    raw_datasets = {}
    for split, file_path in data_files.items():
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found. Skipping {split} split.")
            # Create an empty dataset for the split if file is not found
            raw_datasets[split] = Dataset.from_dict({"tokens": [], "ner_tags": []})
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        all_tokens = []
        all_ner_tags = []
        current_tokens = []
        current_ner_tags = []
        
        for line in lines:
            line = line.strip()
            if line: # Non-empty line, part of a sentence
                parts = line.split('\t')
                if len(parts) == 2:
                    current_tokens.append(parts[0])
                    current_ner_tags.append(parts[1])
                else:
                    print(f"Warning: Skipping malformed line in {file_path}: '{line}'")
            else: # Empty line, end of a sentence
                if current_tokens: # Only add if sentence is not empty
                    all_tokens.append(current_tokens)
                    all_ner_tags.append(current_ner_tags)
                current_tokens = []
                current_ner_tags = []
        
        # Add the last sentence if the file doesn't end with a blank line
        if current_tokens:
            all_tokens.append(current_tokens)
            all_ner_tags.append(current_ner_tags)

        raw_datasets[split] = Dataset.from_dict({"tokens": all_tokens, "ner_tags": all_ner_tags})
    
    return DatasetDict(raw_datasets)


def tokenize_and_align_labels(examples, tokenizer):
    """
    Tokenizes input texts and aligns labels to subwords for NER.
    Handles potential mismatches between word tokens and subword tokens.
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        # DataCollator will handle padding dynamically, so no padding here
    )
    labels = []
    for i, tags_list_str in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i) # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None: # Special tokens (like CLS, SEP)
                label_ids.append(-100) # -100 means ignore in loss calculation
            elif word_idx != previous_word_idx: # Start of a new word
                # Check bounds to prevent IndexError if word_idx somehow exceeds actual tags_list_str length
                if word_idx < len(tags_list_str):
                    label_ids.append(label_to_id[tags_list_str[word_idx]])
                else:
                    label_ids.append(-100) # Should ideally not happen with correct CoNLL
            else: # Continuation of the same word (subword token)
                if word_idx < len(tags_list_str):
                    current_tag_name = tags_list_str[word_idx]
                    # If the original word's tag was 'B-ENTITY', subsequent subwords get 'I-ENTITY'
                    # If it was 'I-ENTITY' or 'O', they keep their original tag
                    if current_tag_name.startswith("B-"):
                        label_ids.append(label_to_id[f"I-{current_tag_name[2:]}"])
                    else:
                        label_ids.append(label_to_id[current_tag_name])
                else:
                    label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(axis=2) # Get the predicted label ID for each token

    # Convert predicted and true label IDs back to their string names for seqeval
    # Also, remove tokens that should be ignored in the loss (-100)
    true_labels = [[id_to_label[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [id_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Filter out empty lists that might occur if a batch has no relevant labels after filtering
    # This prevents errors in seqeval if it receives empty lists
    filtered_true_labels = []
    filtered_true_predictions = []
    for t_l, t_p in zip(true_labels, true_predictions):
        # Only include if there are actual labels and predictions to compare
        if t_l and t_p and len(t_l) == len(t_p): # Ensure lengths match for seqeval
            filtered_true_labels.append(t_l)
            filtered_true_predictions.append(t_p)

    if not filtered_true_labels:
        # If no valid labels or predictions after filtering, return zeros
        print("Warning: No valid labels for metric computation in this batch.")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

    # Calculate standard NER metrics
    precision = precision_score(filtered_true_labels, filtered_true_predictions)
    recall = recall_score(filtered_true_labels, filtered_true_predictions)
    f1 = f1_score(filtered_true_labels, filtered_true_predictions)

    # Calculate overall accuracy (token-level accuracy)
    all_predictions_flat = [p for sublist in filtered_true_predictions for p in sublist]
    all_labels_flat = [l for sublist in filtered_true_labels for l in sublist]
    correct_predictions = sum(1 for p_val, l_val in zip(all_predictions_flat, all_labels_flat) if p_val == l_val)
    accuracy = correct_predictions / len(all_predictions_flat) if all_predictions_flat else 0.0

    print("\n--- Evaluation Report ---")
    try:
        # Print a detailed classification report for each entity type
        report_str = classification_report(filtered_true_labels, filtered_true_predictions, digits=4)
        print(report_str)
    except Exception as e:
        print(f"Error generating classification report: {e}")
    print("-------------------------\n")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def main():
    # Create output and log directories if they don't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGGING_DIR, exist_ok=True)

    # Load dataset
    print(f"Loading data from {DATA_DIR}...")
    raw_datasets = load_conll_data(DATA_DIR)
    
    # Basic check to ensure datasets are loaded and not empty
    if "train" not in raw_datasets or len(raw_datasets["train"]) == 0:
        print(f"Error: Training dataset not found or is empty in {DATA_DIR}.")
        print("Please ensure 'train.conll' exists and contains data.")
        return
    if "validation" not in raw_datasets or len(raw_datasets["validation"]) == 0:
        print(f"Error: Validation dataset not found or is empty in {DATA_DIR}.")
        print("Please ensure 'val.conll' exists and contains data.")
        return
    if "test" not in raw_datasets or len(raw_datasets["test"]) == 0:
        print(f"Error: Test dataset not found or is empty in {DATA_DIR}.")
        print("Please ensure 'test.conll' exists and contains data.")
        return


    print("Dataset loaded successfully:")
    print(raw_datasets)
    print(f"Example from train split: {raw_datasets['train'][0]}")

    # Load tokenizer
    print(f"Loading tokenizer {MODEL_CHECKPOINT}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    # Apply tokenization and label alignment
    print("Tokenizing and aligning labels...")
    tokenized_datasets = raw_datasets.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer),
        batched=True,
        remove_columns=raw_datasets["train"].column_names, # Remove original 'tokens' and 'ner_tags'
        desc="Tokenizing and aligning labels",
    )
    print("Tokenization and alignment complete.")
    print(tokenized_datasets)

    # Load model
    print(f"Loading model {MODEL_CHECKPOINT}...")
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(LABEL_NAMES),
        id2label=id_to_label,
        label2id=label_to_id,
    )
    print("Model loaded.")

    # Initialize Data Collator (for dynamic padding)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Define TrainingArguments (corrected eval_strategy and adjusted hyperparameters)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch", # CORRECTED: Changed from evaluation_strategy
        learning_rate=1e-5, # ADJUSTED: Smaller learning rate
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10, # ADJUSTED: Increased number of epochs
        weight_decay=0.01,
        logging_dir=LOGGING_DIR,
        logging_strategy="epoch", # Log evaluation metrics at the end of each epoch
        save_strategy="epoch",    # Save checkpoint at the end of each epoch
        load_best_model_at_end=True, # Load the best model (based on metric_for_best_model) at the end of training
        metric_for_best_model="f1", # The metric to monitor for best model selection
        report_to="none", # You can change this to "tensorboard" or "wandb" later for visualization
        fp16=True, # Enable mixed precision training for faster training on compatible GPUs (if available)
        push_to_hub=False, # Set to True if you want to push your model to Hugging Face Hub
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer, # Pass tokenizer to Trainer for proper tokenization within it
        data_collator=data_collator,
        compute_metrics=compute_metrics, # Pass the metric computation function
    )

    # --- Start Training! ---
    print("\n--- Starting Model Training ---")
    train_result = trainer.train()
    print("\n--- Training Complete ---")

    # Save the fine-tuned model and tokenizer
    print(f"Saving fine-tuned model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR) # Save tokenizer alongside the model

    # Evaluate on the test set after training
    print("\n--- Evaluating on Test Set ---")
    test_results = trainer.evaluate(tokenized_datasets["test"])
    print(f"Test Set Evaluation Results: {test_results}")


if __name__ == "__main__":
    main()