import pandas as pd
import os
import json
import re
import pytesseract
from PIL import Image

# Configure Tesseract executable path if it's not in your system's PATH
# If Tesseract.exe is in "C:\Program Files\Tesseract-OCR\tesseract.exe", uncomment and adjust path below:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# For macOS/Linux, Tesseract is typically in PATH after installation, so this line might not be needed.

def preprocess_amharic_text(text):
    """
    Performs basic preprocessing on Amharic text.
    This is a starting point and can be expanded for more sophisticated Amharic NLP.
    """
    if not isinstance(text, str):
        return None
    # Remove extra spaces and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    # You might add more specific Amharic normalization here, e.g.,
    # handling different forms of characters, removing punctuation specific to Amharic, etc.
    return text

def extract_text_from_image(image_path):
    """
    Extracts Amharic (and English) text from an image using Tesseract OCR.
    Assumes amh.traineddata and eng.traineddata are installed.
    """
    if not image_path or not os.path.exists(image_path):
        return None
    try:
        img = Image.open(image_path)
        # Use both Amharic and English language packs for better coverage
        text = pytesseract.image_to_string(img, lang='amh+eng')
        return text.strip() if text else None
    except Exception as e:
        print(f"Error extracting text from image {image_path}: {e}")
        return None

def load_raw_data(raw_data_path='data/raw/all_raw_telegram_messages.json'):
    """Loads raw scraped data from a JSON file."""
    if not os.path.exists(raw_data_path):
        print(f"Raw data file not found: {raw_data_path}. Please run scraper.py first.")
        return []
    with open(raw_data_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def clean_and_structure_data(raw_data):
    """
    Cleans and structures the raw message data into a DataFrame.
    Separates metadata from content and performs OCR for images.
    """
    processed_records = []
    for record in raw_data:
        cleaned_text = preprocess_amharic_text(record.get('text'))
        image_text = None

        # If an image path exists, attempt OCR
        if record.get('image_local_path'):
            image_text = extract_text_from_image(record['image_local_path'])

        # Combine original text and image text if both exist
        # Prioritize original text, but add image text if available and not redundant
        combined_text = cleaned_text if cleaned_text else ""
        if image_text and image_text not in combined_text: # Avoid duplicating if image text is already in message text
            if combined_text:
                combined_text += " " + image_text
            else:
                combined_text = image_text
        
        # If after all processing, combined_text is empty, set to None
        if not combined_text:
            combined_text = None

        processed_records.append({
            'message_id': record.get('message_id'),
            'channel_name': record.get('channel_name'),
            'sender_id': record.get('sender_id'),
            'date': record.get('date'),
            'text_raw': record.get('text'),
            'text_cleaned': cleaned_text, # Original text cleaned
            'image_local_path': record.get('image_local_path'), # Path to image
            'image_text_extracted': image_text, # Text extracted from image
            'combined_content_for_ner': combined_text, # Combined text for NER
            'views': record.get('views'),
            'media_type': record.get('media_type')
        })
    return pd.DataFrame(processed_records)

def save_processed_data(df, output_path='data/processed/cleaned.parquet'):
    """Saves the processed DataFrame to a parquet file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Processed data saved to {output_path}")

def main():
    raw_data = load_raw_data()
    if raw_data:
        df_processed = clean_and_structure_data(raw_data)
        # Filter out messages with no meaningful text after cleaning or OCR
        df_processed = df_processed.dropna(subset=['text_cleaned', 'image_text_extracted'], how='all').reset_index(drop=True)
        save_processed_data(df_processed)
        print(f"Preprocessing complete. Total messages processed (including OCR): {len(df_processed)}")
    else:
        print("No raw data to process.")

if __name__ == '__main__':
    main()