# 🛍️ Amharic E-commerce Data Extractor

## 📌 Project Overview

The **Amharic E-commerce Data Extractor** is an end-to-end pipeline for transforming unstructured Telegram e-commerce posts (text and images) into structured, business-ready data using **Named Entity Recognition (NER)**. It extracts key entities — **Product**, **Price**, **Location**, **Brand**, **Size**, and **Contact** — to power EthioMart, a centralized Amharic e-commerce hub.

The project addresses the challenge of decentralized commerce on Telegram by scraping, processing, labeling, and analyzing messages from various channels. It includes fine-tuning and comparing transformer models for Amharic NER, model interpretability, and vendor analytics for micro-lending.

---

## 🗂️ Project Structure

```bash
├── .github/
│   └── workflows/                  # CI/CD & model evaluation workflows
├── configs/
│   └── scraping_config.yaml        # Telegram API config (IDs, target channels)
├── data/
│   ├── raw/                       # Raw scraped messages & images
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
├── models/
│   ├── *_finetuned/                # Fine-tuned model checkpoints (ignored by git)
│   └── model_cards/                # Evaluation results, model cards
├── reports/
│   └── vendor_scorecard.csv        # Vendor analytics output
├── src/
│   ├── analytics/
│   │   └── vendor_scoring.py       # Vendor analytics & scorecard
│   ├── data_pipeline/
│   │   ├── scraper.py              # Telegram scraper
│   │   ├── preprocessor.py         # OCR & cleaning logic
│   │   └── labeling_tool.py        # Sampling & CoNLL conversion
│   ├── ml_pipeline/
│   │   ├── training.py             # Model training script
│   │   ├── evaluation.py           # Model evaluation & comparison
│   │   └── interpretability.py     # LIME-based model interpretability
│   └── utils/
├── requirements.txt
└── README.md
```

---

## ✅ Task 1: Data Ingestion & Preprocessing

**Goal:** Scrape Telegram messages & images, perform OCR, and prepare unified text for NER.

**How to Run:**

1. Configure `configs/scraping_config.yaml` with your Telegram API credentials and channel list.
2. Scrape Telegram data:
   ```bash
   python src/data_pipeline/scraper.py
   ```
3. Run preprocessing (OCR + cleaning):
   ```bash
   python src/data_pipeline/preprocessor.py
   ```
   - Output: `data/processed/cleaned.parquet`

---

## ✅ Task 2: Manual Labeling & CoNLL Conversion

**Goal:** Create a high-quality, human-annotated NER dataset in CoNLL format with BIO tagging.

**How to Run:**

1. Sample messages for annotation:
   ```bash
   python src/data_pipeline/labeling_tool.py --action sample --num_samples 300
   ```
2. Annotate messages in `data/labeled/raw_for_annotation/messages_for_manual_labeling.json`.
3. Convert to CoNLL format:
   ```bash
   python src/data_pipeline/labeling_tool.py --action convert --annotated_file data/labeled/raw_for_annotation/messages_for_manual_labeling.json
   ```
   - Output: `train.conll`, `val.conll`, `test.conll`

---

## ✅ Task 3: Model Fine-Tuning

**Goal:** Fine-tune transformer models (mBERT, DistilBERT, XLM-R) for Amharic NER using the labeled CoNLL data.

**How to Run:**

```bash
python src/ml_pipeline/training.py
```

- Configure the model checkpoint in the script to switch between models.
- Outputs are saved in `models/*_finetuned/` (ignored by git).

---

## ✅ Task 4: Model Evaluation & Comparison

**Goal:** Evaluate and compare fine-tuned models on the test set using F1, precision, and recall. Select the best model for production.

**How to Run:**

```bash
python src/ml_pipeline/evaluation.py
```

- Outputs a comparison table and saves results to `models/model_cards/evaluation_results.csv`.

---

## ✅ Task 5: Model Interpretability (LIME)

**Goal:** Use LIME to explain which words most influence the model's prediction of entity presence in Amharic sentences.

**How to Run:**

```bash
python src/ml_pipeline/interpretability.py
```

- Prints word importances for each sample sentence and entity type.
- You can change the entity type in the script to analyze different entities.

---

## ✅ Task 6: Vendor Analytics & Scorecard

**Goal:** Combine NER results and message metadata to compute business metrics for each vendor/channel (posting frequency, average views, average price, top post, lending score).

**How to Run:**

```bash
python src/analytics/vendor_scoring.py
```

- Outputs a summary table and saves to `reports/vendor_scorecard.csv`.

---

## 🚀 Next Steps / Extending the Project

- Improve NER accuracy with more labeled data.
- Add more business metrics or visualizations.
- Integrate with a live dashboard or database.
- Expand interpretability to more entity types or use SHAP (with custom wrappers).

---

## 🧩 Dependencies

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## 👥 Contributors

This project is built collaboratively for improving Ethiopian e-commerce accessibility and intelligence. For contribution guidelines and technical documentation, please refer to the respective module READMEs.

---

## 📄 License

Licensed under MIT. See `LICENSE` file for details.
