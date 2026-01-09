# Word2Vec on Federal Reserve Communications

This repository contains the final project for a Big Data course, implementing **Word2Vec embeddings trained on Federal Reserve communications** using **Apache Spark** and **Apache Zeppelin**.

The goal of the project is to evaluate whether distributional word embeddings can capture meaningful **monetary policy semantics** from official Fed text, using analogy tests and nearest-neighbor analysis rather than downstream prediction tasks.

---

## Dataset

The corpus consists of cleaned and filtered Federal Reserve communications, including:

* **FOMC Statements** (1994–present)
* **FOMC Minutes** (1994–present)
* **Fed Chair Speeches** (Bernanke, Yellen, Powell)
* **Press Conference Transcripts** (post-2011)

After preprocessing, the dataset contains approximately **350 policy paragraphs (~1M tokens)**.

Processed data are stored in **Parquet format** for efficient reuse and reproducibility.

---

## Preprocessing & Tokenization

Generic tokenization was insufficient for this domain, so a **Fed-specific pipeline** was implemented:

* **Lowercasing** and punctuation normalization (underscores preserved)
* **Manual phrase merging** for policy concepts:

  * `balance_sheet`, `asset_purchases`, `forward_guidance`, `federal_funds_rate`, etc.
* **Acronym preservation** (e.g., `qe`, `iorb`, `on_rrp`)
* **Numeric normalization**:

  * Percentages → `x_percent`
  * Rate moves → `x_basis_points`
* Filtering to retain:

  * merged phrases
  * acronyms
  * alphabetic tokens
  * normalized numeric tokens

All evaluation terms (anchors and analogies) are drawn **directly from the tokenized vocabulary**, ensuring exact alignment (no casing or spacing mismatches).

---

## Model Training

Word2Vec models were trained using **Spark MLlib**, varying key hyperparameters:

* `vectorSize` ∈ {100, 200, 300}
* `windowSize` ∈ {5, 10, 15}
* `minCount` ∈ {3, 5, 10}

Training was performed in Apache Zeppelin with fixed random seeds for comparability.

### Model Selection

Models were compared using:

* **Vocabulary coverage** of predefined anchor terms (sanity check)
* **Analogy performance** (average cosine similarity across test equations)

The final model was selected based on **quantitative analogy performance**, not vocabulary size alone.

**Best model:**
`vectorSize = 100`, `windowSize = 15`, `minCount = 5`

---

## Evaluation

### Analogy Tests

Analogies follow the standard Word2Vec vector arithmetic form:

```
A − B + C ≈ D
```

Examples include:

* `tapering − asset_purchases + balance_sheet ≈ runoff`
* `quantitative_easing − asset_purchases + balance_sheet ≈ normalization`
* `tightening − hikes + cuts ≈ easing`

Performance was measured via **cosine similarity** of top-k nearest neighbors, excluding query terms.

### Nearest Neighbors

Nearest-neighbor analysis was conducted for:

* Macro variables (`inflation`, `labor_market`)
* Policy tools (`tapering`, `balance_sheet`)
* Policy stance terms (`tightening`, `easing`)

Results demonstrate that the model captures meaningful **policy-relevant context**, while also revealing known Word2Vec limitations.

---

## Key Findings

* **Context window size mattered more than embedding dimensionality** for this corpus
* Policy tools and balance-sheet concepts were captured more cleanly than abstract stance terms
* Some intuitively plausible analogies failed due to:

  * regime mixing across time
  * polysemy (single vector per word)
  * distributional similarity (antonyms sharing context)

These failure modes are consistent with known properties of Word2Vec models.

---

## Limitations & Future Work

* The corpus size, while coherent, is relatively small for Word2Vec, limiting semantic resolution
* Word2Vec collapses multiple meanings of a term into a single vector
* No market data are incorporated; embeddings reflect **linguistic structure only**

Future extensions could include:

* Time-segmented embeddings to study regime shifts
* Linking embeddings to market reactions (rates, volatility) in a separate analysis
* Comparison with contextual or transformer-based embeddings

---

## Repository Structure

```
.
├── src/fed_word2vec/   # Python package (pip installable)
│   ├── __init__.py
│   ├── cli.py          # Command-line interface
│   ├── config.py       # Configuration management
│   ├── downloader.py   # Fed transcript scraping & downloading
│   └── text_cleaner.py # HTML parsing & text filtering
├── notebooks/          # Orchestration & demo notebooks
│   ├── *.zpln          # Zeppelin notebooks (Spark/Scala Word2Vec training)
│   └── *.ipynb         # Jupyter notebooks (Python data pipeline)
├── data/               # Raw downloads and processed text (gitignored)
├── model/              # Saved Word2Vec models (gitignored)
├── config.yaml         # Pipeline configuration
├── pyproject.toml      # Python package definition
├── report/             # Final project report (PDF)
└── README.md
```

> **Note:** `data/` and `model/` directories are excluded from version control via `.gitignore`. Run the download/clean pipeline to populate them.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/7josh7/fed-word2vec.git
cd fed-word2vec

# Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install in editable mode
pip install -e .
```

---

## CLI Usage

The package provides a command-line interface for downloading and cleaning Fed documents:

```bash
# Show help
fed-word2vec --help

# Initialize a config file (optional, one is already included)
fed-word2vec init

# Download all Fed documents (FOMC + speeches)
fed-word2vec download

# Download only FOMC documents
fed-word2vec download fomc

# Clean HTML files to text
fed-word2vec clean

# Clean with removed-lines log for debugging
fed-word2vec clean --save-removed
```

### Quick Smoke Test

To verify the installation works without downloading the full corpus:

```bash
# This will attempt to connect and parse the Fed website
fed-word2vec download fomc
# Then cancel (Ctrl+C) after a few files download
```

---

## Requirements

### For Data Pipeline (Python)
* Python 3.10+
* Dependencies installed via `pip install -e .`

### For Model Training (Spark)
* Apache Spark
* Apache Zeppelin
* Scala (Spark MLlib)

---

## Notes

This project focuses on **interpretability and semantic evaluation**, in line with course objectives, rather than downstream prediction or economic modeling.

---

