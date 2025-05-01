# Sentiment-Analysis
📦 Product Review Sentiment Analyzer

This project analyzes customer reviews to determine the sentiment (positive, negative, or neutral) using **VADER** (Valence Aware Dictionary for sEntiment Reasoning), a rule-based sentiment analysis tool from **NLTK**. It supports batch processing from CSV files, single review analysis, interactive CLI usage, and a full-featured GUI application.

🔍 Key Features

- Analyze single reviews or entire datasets
- GUI application with sentiment bars and visualizations
- Accuracy, precision, F1-score reporting (if ground truth is available)
- Export results and plots (CSV, Excel, PNG)
- Auto-detects review columns from uploaded CSVs

🤖 Machine Learning Algorithm Used

**Algorithm:** VADER Sentiment Analysis (from NLTK)

- **Type:** Rule-based Natural Language Processing
- **No training required** – uses a lexicon of words with associated sentiment scores.
- Handles:
  - Capitalization and punctuation (e.g., "LOVE!!!")
  - Degree modifiers (e.g., "very good")
  - Negations (e.g., "not bad")

**This is not a supervised ML model.** Instead, it uses pre-defined rules and word scores to assign sentiment labels.

🧪 Project Workflow

  1. **Generate Sample Data**
You can generate realistic review data with labeled sentiment using:
```bash
python generate_sample_data.py
```

This will create a CSV file `sample_product_reviews.csv` with synthetic reviews.

  2. **Run Sentiment Analysis**

#### Option 1: Command Line

**Batch CSV analysis:**
```bash
python main.py --file sample_product_reviews.csv
```

**Single review:**
```bash
python main.py --text "This product is amazing!"
```

**Interactive mode:**
```bash
python main.py
```

  Option 2: GUI Mode

Launch the GUI:
```bash
python gui.py
```

- Analyze individual reviews
- Upload CSV files and visualize results
- Export outcomes and performance metrics


📦 Installation

1. Clone or download this repository.
2. Create a virtual environment (optional but recommended).
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

📁 File Overview

| File | Description |
|------|-------------|
| `generate_sample_data.py` | Generates labeled synthetic reviews |
| `main.py` | CLI-based sentiment analyzer |
| `gui.py` | Tkinter-based GUI application |
| `sentiment_analyzer.py` | Core logic using VADER |
| `requirements.txt` | Lists required Python libraries |
| `sample_product_reviews.csv` | Example data (auto-generated) |

📊 Output

- Sentiment breakdown: Positive / Neutral / Negative
- Sentiment score bars
- Compound score distributions
- Confusion matrix (if labeled data is used)
- Accuracy, precision, F1 score

📎 Example

```bash
$ python main.py --text "This phone is terrible. It broke in a week."
Sentiment: negative
Scores: Positive=0.000, Neutral=0.456, Negative=0.544
Compound Score: -0.648
```

✅ Requirements

- Python 3.7+
- Libraries: `nltk`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`
