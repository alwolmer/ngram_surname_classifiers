# Mapping Ancestry through Surnames: Machine Learning Approaches Applied to Brazilian Data

This repository contains the full codebase, data pipeline, and trained models associated with the research article **"Mapping Ancestry through Surnames: Machine Learning Approaches Applied to Brazilian Data"**, submitted to the KDMiLe 2025 conference.

The study proposes and compares character-based n-gram models and graph-based classifiers (LIGA) to infer ethnic ancestry from historical surnames in Brazil, using a large-scale, scraped dataset from **Registro Nacional de Estrangeiros (RNE)** entries.

---

## 📁 Repository Structure

ngram_surname_classifiers/
│
├── data/
│ ├── raw/ # Original scraped RNE records
│ ├── intermediate/ # Cleaned data, extracted surnames, ground truth
│ └── outputs/ # Serialized trained models and prediction outputs
│
├── scripts/
│ ├── scraper/ # Web scraping scripts for RNE data
│ │ └── rne_scraper.py
│ ├── pipeline/ # Data preprocessing and cleaning notebooks
│ │ └── preprocessing_pipeline.ipynb
│ ├── models/ # Model definitions (Cavnar-Trenkle, LIGA, Ensemble)
│ │ ├── ngram_classifier.py
│ │ └── liga_classifier.py
│ └── evaluation/ # Evaluation logic, stats tests, plots
│ └── evaluation_pipeline.ipynb


---

## 🚀 Getting Started

### Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

🧠 Models

The project implements and compares:

    N-gram classifiers (Cavnar & Trenkle inspired)

    LIGA (Language Identification, Graph-based Approach classifier)

    Meta-ensemble model combining predictions of the above


All models are trained to predict six ancestry classes: Iberian, Italian, Japanese, Germanic, Arabic, and Eastern European.
📊 Evaluation

Evaluation includes:

    Cross-validation metrics (macro-F1, per-class F1)

    Statistical tests (paired one-tailed t-tests, Cohen’s d)

    Visual summaries with margin of error

📂 Data

All records were extracted from the public archive:

    Arquivo Público do Estado de São Paulo - Delegacia de Estrangeiros

Scraped dataset contains over 1 million raw records from the RNE archive.
