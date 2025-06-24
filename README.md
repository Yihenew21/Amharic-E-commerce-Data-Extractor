# 🛍️ Amharic E-commerce Data Extractor

## 📌 Project Overview

The **Amharic E-commerce Data Extractor** is a pipeline for transforming unstructured Telegram e-commerce posts (both text and images) into a structured format through **Named Entity Recognition (NER)**. It extracts key business entities — such as **Product**, **Price**, **Location**, **Brand**, **Size**, and **Contact** — and organizes them in a centralized format for **EthioMart**, an Amharic e-commerce hub.

The project addresses the challenge of decentralized commerce on Telegram by scraping, processing, and labeling messages from various channels. It lays the groundwork for fine-tuning **Large Language Models (LLMs)** specifically for Amharic NER tasks.

---

## 🗂️ Project Structure

```bash
├── .github/
│   └── workflows/                  # CI/CD pipelines (To be added)
├── configs/
│   └── scraping_config.yaml        # Telegram API config (IDs, target channels)
├── data/
│   ├── raw/
│   │   ├── telegram/
│   │   │   └── [channel_name]/     # Per-channel scraped messages + images
│   │   └── all_raw_telegram_messages.json
│   ├── processed/
│   │   └── cleaned.parquet         # Preprocessed data (OCR + text)
│   └── labeled/
│       ├── raw_for_annotation/
│       │   └── messages_for_manual_labeling.json
│       ├── v1/
│       │   ├── train.conll
│       │   ├── val.conll
│       │   └── test.conll
│       └── README.md               # Labeling rules and entity definitions
├── ml_pipeline/
│   ├── training.py                 # Model training script (TBD)
│   └── evaluation.py               # Model evaluation script (TBD)
├── src/
│   └── data_pipeline/
│       ├── scraper.py              # Telegram scraper
│       ├── preprocessor.py         # OCR & cleaning logic
│       └── labeling_tool.py        # Sampling & CoNLL conversion
├── .gitignore
├── requirements.txt
└── README.md                       # You're reading it
```

---

## ✅ Task 1: Data Ingestion & Preprocessing

### 🎯 Objective

Build a robust ingestion system that:

- Scrapes Telegram messages & media (images),
- Performs OCR on images,
- Combines content for NER preprocessing.

### ⚙️ Components

#### 🔹 `scraper.py`

- Connects to Telegram using `Telethon`.
- Reads credentials and channels from `configs/scraping_config.yaml`.
- Downloads message metadata + image attachments.
- Outputs:
  - Raw per-channel data: `data/raw/telegram/[channel_name]/`
  - Unified JSON: `data/raw/all_raw_telegram_messages.json`

#### 🔹 `preprocessor.py`

- Loads raw messages.
- Cleans Amharic text (spacing, newline issues).
- Performs **OCR** on downloaded images using `Pytesseract` with `amh+eng` models.
- Merges original + OCR text into `combined_content_for_ner`.
- Outputs:
  - Preprocessed Parquet file: `data/processed/cleaned.parquet`

### 🚧 Challenges & Solutions

- **Telegram Auth:** Handled `Telethon` session persistence.
- **Image Management:** Stored local paths for images, handled naming conflicts.
- **OCR Accuracy:** Configured Tesseract to support Amharic (`amh.traineddata`) alongside English.

### ▶️ How to Run Task 1

1. **Configure**

   ```yaml
   # configs/scraping_config.yaml
   api_id: YOUR_API_ID
   api_hash: YOUR_API_HASH
   channels:
     - ecommerce_channel_1
     - ecommerce_channel_2
   ```

2. **Scrape Telegram Data**

   ```bash
   python src/data_pipeline/scraper.py
   ```

3. **Run Preprocessing (OCR + Cleaning)**
   ```bash
   python src/data_pipeline/preprocessor.py
   ```

---

## ✅ Task 2: Manual Labeling & CoNLL Conversion

### 🎯 Objective

Create a high-quality, **human-annotated NER dataset** in **CoNLL** format with BIO tagging for fine-tuning a custom NER model.

### ⚙️ Components

#### 🔹 `labeling_tool.py`

- **Sampling Mode**

  ```bash
  python src/data_pipeline/labeling_tool.py --action sample --num_samples 300
  ```

  - Extracts 300 random entries from `cleaned.parquet`.
  - Saves to: `data/labeled/raw_for_annotation/messages_for_manual_labeling.json`

- **Annotation Format**

  ```json
  {
    "message_id": 12345,
    "combined_content_for_ner": "Sample Amharic message...",
    "annotations": [
      {
        "entity": "PRICE",
        "text": "1500 ብር",
        "start": 10,
        "end": 17
      }
    ]
  }
  ```

- **Conversion Mode**
  ```bash
  python src/data_pipeline/labeling_tool.py --action convert --annotated_file data/labeled/raw_for_annotation/messages_for_manual_labeling.json
  ```
  - Converts annotated messages into token-level **BIO CoNLL format**.
  - Splits into: `train.conll`, `val.conll`, `test.conll`

#### 📘 `data/labeled/README.md`

- **Entity Definitions:** Detailed labeling rules for each of the six target entities.
- **Guidance:** Examples and best practices for consistency across annotators.
- ⚠️ **This file must be created and updated manually.**

### 🚧 Challenges & Solutions

- Designed simple, intuitive JSON structure for annotators.
- Wrote robust tokenizer-to-BIO converter to handle annotation alignment.
- Solved minor bugs (e.g., multi-line print syntax, file path mismatches).

---

### ▶️ How to Run Task 2

1. **Sample Messages**

   ```bash
   python src/data_pipeline/labeling_tool.py --action sample --num_samples 300
   ```

2. **Annotate Messages**

   - Open: `data/labeled/raw_for_annotation/messages_for_manual_labeling.json`
   - Add `annotations` for each message manually.

3. **Convert to CoNLL Format**

   ```bash
   python src/data_pipeline/labeling_tool.py --action convert --annotated_file data/labeled/raw_for_annotation/messages_for_manual_labeling.json
   ```

4. **Write Labeling Guidelines**
   - Add instructions to: `data/labeled/README.md`

---

## 🚀 Next Steps: Task 3 – Model Fine-Tuning

With the pipeline and labeled data ready, the project is now set to move into **Task 3: Fine-tuning the NER model** using the prepared CoNLL data. This includes training scripts, hyperparameter tuning, and evaluation, to be implemented in the `ml_pipeline/` directory.

---

## 🧩 Dependencies

Install project dependencies with:

```bash
pip install -r requirements.txt
```

---

## 👥 Contributors

This project is built collaboratively for improving Ethiopian e-commerce accessibility and intelligence. For contribution guidelines and technical documentation, please refer to the respective module READMEs.

---

## 📄 License

Licensed under MIT. See `LICENSE` file for details.
