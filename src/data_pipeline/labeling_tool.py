import pandas as pd
import os
import json
import re
from sklearn.model_selection import train_test_split # Added for splitting data

def load_processed_data(input_path='data/processed/cleaned.parquet'):
    """Loads processed data from a parquet file."""
    if not os.path.exists(input_path):
        print(f"Processed data file not found: {input_path}. Please run preprocessor.py first.")
        return pd.DataFrame()
    return pd.read_parquet(input_path)

def sample_for_labeling(df, num_samples=50, output_dir='data/labeled/raw_for_annotation'):
    """
    Samples messages for manual labeling and saves them to a JSON file.
    Only includes 'message_id' and 'combined_content_for_ner'.
    """
    if df.empty:
        print("No data to sample.")
        return

    df_eligible = df.dropna(subset=['combined_content_for_ner']).copy()

    if len(df_eligible) < num_samples:
        print(f"Warning: Not enough messages ({len(df_eligible)}) to sample {num_samples}. Sampling all available.")
        sampled_df = df_eligible
    else:
        sampled_df = df_eligible.sample(n=num_samples, random_state=42)

    labeling_data = sampled_df[['message_id', 'combined_content_for_ner']].to_dict(orient='records')

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'messages_for_manual_labeling.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(labeling_data, f, ensure_ascii=False, indent=4)
    print(f"Sampled {len(labeling_data)} messages for labeling saved to {output_path}")
    print("\n--- Next Step: Manual Annotation ---")
    print(f"Please manually annotate the entities in '{output_path}'.")
    # CORRECTED LINE BELOW: Using triple quotes
    print("""Add an 'annotations' list to each message entry, containing dictionaries with 'entity', 'text', 'start', and 'end'.
Example: {"entity": "PRODUCT", "text": "...", "start": ..., "end": ...}""")
    print("Once annotated, run this script again with the 'convert' action to generate CoNLL files.")


def text_to_tokens_and_tags(text, annotations):
    """
    Converts a text and its annotations into a list of (token, tag) pairs in BIO format.
    Assumes simple whitespace tokenization for Amharic. More advanced tokenization
    would be needed for production.
    """
    tokens = text.split()
    # Initialize tags with 'O' (Outside)
    tags = ['O'] * len(tokens)

    # Calculate token spans to map character offsets to token indices
    token_spans = []
    current_char = 0
    for token in tokens:
        start = text.find(token, current_char)
        end = start + len(token)
        token_spans.append((start, end))
        current_char = end + 1 # Update for next search, handling multiple spaces

    for ann in annotations:
        ann_start = ann['start']
        ann_end = ann['end']
        ann_entity = ann['entity']

        # Find which tokens overlap with the annotation span
        affected_token_indices = []
        for i, (token_start, token_end) in enumerate(token_spans):
            # Check for overlap: [token_start, token_end) and [ann_start, ann_end)
            if max(token_start, ann_start) < min(token_end, ann_end):
                affected_token_indices.append(i)

        if not affected_token_indices:
            # print(f"Warning: Annotation '{ann['text']}' at {ann_start}-{ann_end} did not align with any tokens. Text: '{text}'")
            continue # Skip if annotation doesn't align with tokens

        for i, token_idx in enumerate(affected_token_indices):
            if i == 0:
                tags[token_idx] = f'B-{ann_entity}'
            else:
                tags[token_idx] = f'I-{ann_entity}'
    return list(zip(tokens, tags))

def convert_to_conll_format(annotated_data_path, output_dir='data/labeled/v1',
                             train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Converts manually annotated JSON data into CoNLL format (train, val, test splits).
    """
    if not os.path.exists(annotated_data_path):
        print(f"Annotated data file not found: {annotated_data_path}. Please annotate first.")
        return

    with open(annotated_data_path, 'r', encoding='utf-8') as f:
        annotated_messages = json.load(f)

    conll_sentences = []
    for msg in annotated_messages:
        text = msg.get('combined_content_for_ner')
        annotations = msg.get('annotations', [])
        if text:
            tokens_and_tags = text_to_tokens_and_tags(text, annotations)
            if tokens_and_tags:
                conll_sentences.append(tokens_and_tags)

    if not conll_sentences:
        print("No valid annotated sentences found to convert to CoNLL.")
        return

    # Split data
    train_val, test_data = train_test_split(conll_sentences, test_size=test_ratio, random_state=42)
    train_data, val_data = train_test_split(train_val, test_size=val_ratio/(train_ratio + val_ratio), random_state=42)

    splits = {
        'train.conll': train_data,
        'val.conll': val_data,
        'test.conll': test_data
    }

    os.makedirs(output_dir, exist_ok=True)
    for filename, data in splits.items():
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            for sentence in data:
                for token, tag in sentence:
                    f.write(f"{token}\t{tag}\n")
                f.write("\n") # Blank line separates sentences
        print(f"Generated {len(data)} sentences for {filename} at {output_path}")

    print("\n--- CoNLL Conversion Complete ---")
    print(f"Please review the generated CoNLL files in '{output_dir}'.")
    print("Remember to update 'data/labeled/README.md' with your labeling guidelines.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Tool for sampling and converting data for NER labeling.")
    parser.add_argument('--action', type=str, default='sample',
                        choices=['sample', 'convert'],
                        help="Action to perform: 'sample' messages for labeling, or 'convert' annotated data to CoNLL.")
    parser.add_argument('--num_samples', type=int, default=50,
                        help="Number of messages to sample for labeling (only with 'sample' action).")
    parser.add_argument('--annotated_file', type=str,
                        default='data/labeled/raw_for_annotation/messages_for_manual_labeling.json',
                        help="Path to the manually annotated JSON file (only with 'convert' action).")

    args = parser.parse_args()

    if args.action == 'sample':
        df = load_processed_data()
        sample_for_labeling(df, args.num_samples)
    elif args.action == 'convert':
        convert_to_conll_format(args.annotated_file)

if __name__ == '__main__':
    main()