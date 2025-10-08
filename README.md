# Emotion-Detection-Task-
Third Assignment of Machine Learning Course - 2024
## Emotion Detection (Persian Text) — Machine Learning Assignment 3

**Author**: Diba Elahi  
**Course**: Machine Learning  
**Assignment**: No. 3

This project builds an emotion detection system for Persian text. It includes preprocessing with `hazm`, feature extraction via TF‑IDF and Word2Vec, and multiple classifiers (Decision Tree, Random Forest, Gradient Boosting, XGBoost, Logistic Regression, SVC). A soft-voting ensemble of Logistic Regression and SVC achieves the best performance.

A detailed discussion is provided in the accompanying report (`Report.pdf`).

### Highlights
- **Language**: Persian (Farsi)
- **Preprocessing**: Normalization, URL/HTML removal, punctuation/digit/emoji removal, tokenization, Persian stopwords filtering
- **Features**:
  - TF‑IDF (1–3 n‑grams, max_features=10000, min_df=5, max_df=0.9)
  - Word2Vec sentence embeddings (size=100, window=5); averaged token vectors
  - Final representation: TF‑IDF sparse matrix horizontally stacked with embeddings
- **Labels**: `ANGRY`, `FEAR`, `HAPPY`, `OTHER`, `SAD`
- **Best Model**: Soft Voting (Logistic Regression + SVC)
  - 5‑fold CV (best fold): ~63.86% accuracy; std ~1.94%
  - Logistic Regression alone (best fold): ~62.34% (std ~1.99%)
  - SVC alone (best fold): ~61.52% (std ~1.04%)

---

## Repository Contents
- `Emotion_Detection_Task.ipynb`: Main notebook with end‑to‑end pipeline
- `Report.pdf`: Project report with methodology, results, and discussion

Expected data files (see Setup):
- `train_data.xlsx` → columns: `sentence`, `emotion`
- `3rdHW_test.csv` → column: `X` (test sentences)

Outputs:
- `result.csv` → predictions for the test set with column `Y` (emotion labels)

---

## Setup

### 1) Environment
- Python 3.10+ is recommended.

Install dependencies (suggested minimal set based on the notebook):
```bash
pip install \
  pandas==2.* numpy==1.24.3 scipy==1.11.4 scikit-learn==1.2.2 \
  hazm==0.10.0 nltk==3.8.1 gensim==4.3.2 fasttext-wheel==0.9.2 \
  python-crfsuite==0.9.10 smart-open==7.* wordcloud==1.* \
  matplotlib==3.* seaborn==0.13.* xgboost==1.* joblib==1.* \
  jupyter
```

### 2) Data Placement
Place your data in the project root:
- `train_data.xlsx` (training data)
- `3rdHW_test.csv` (test data)

Note: The notebook currently references absolute paths like `/train_data.xlsx`, `/3rdHW_test.csv`, and writes `/result.csv`. If you prefer relative paths, update those cells to `./train_data.xlsx`, `./3rdHW_test.csv`, and `./result.csv`.

---

## How to Run

1. Launch Jupyter and open the notebook:
   ```bash
   jupyter notebook Emotion_Detection_Task.ipynb
   ```
2. Run all cells in order.
3. Ensure the training and test files are present as described above.
4. Upon completion, the notebook will write predictions to `result.csv` (by default at the filesystem root `/result.csv`; change to a relative path if desired).

---

## Methodology

### Preprocessing
- Normalize text (`hazm.Normalizer`)
- Remove URLs/HTML
- Remove punctuation, digits, and emojis
- Tokenize (`hazm.WordTokenizer`)
- Remove Persian stopwords (custom list)

### Feature Engineering
- TF‑IDF with 1–3 n‑grams (max_features=10000)
- Train Word2Vec on the combined corpus (train+test sentences) and average token vectors to form sentence embeddings (size=100)
- Concatenate TF‑IDF sparse vectors with dense embeddings

### Models Evaluated
- Decision Tree (GridSearchCV)
- Random Forest (GridSearchCV)
- Gradient Boosting (GridSearchCV)
- XGBoost (GridSearchCV)
- Logistic Regression (5‑fold CV)
- SVC (5‑fold CV)
- Soft Voting Ensemble: Logistic Regression + SVC (GridSearchCV over key hyperparameters; 5‑fold CV evaluation)

### Selected Model
- Soft Voting (LogReg + SVC, probability=True, voting='soft')
- Label encoding/decoding handled with `sklearn.preprocessing.LabelEncoder`

---

## Results (Validation)

- Gradient Boosting: ~57.97% accuracy (weighted F1 ~0.58)
- XGBoost: ~56.45% accuracy (weighted F1 ~0.56)
- Random Forest: ~49.14% accuracy
- Decision Tree: ~40.91% accuracy
- Logistic Regression (best fold): ~62.34% accuracy
- SVC (best fold): ~61.52% accuracy
- Soft Voting Ensemble (best fold): **~63.86% accuracy**, std ~1.94%

Final predictions for the test set are saved to `result.csv` with column `Y`.

---

## Reproducibility Notes
- `numpy` RNG is seeded with `42`.
- Grid/cross‑validation splits are controlled by `random_state=42` where applicable.
- Ensure consistent package versions as specified to match reported results.

---

## Troubleshooting
- If you see file not found errors for `/train_data.xlsx` or `/3rdHW_test.csv`, either:
  - Place files at the filesystem root, or
  - Change notebook paths to relative: `./train_data.xlsx`, `./3rdHW_test.csv`, `./result.csv`.
- If memory issues occur while training, reduce `max_features` in TF‑IDF or lower `n_estimators` for ensemble models.

---

## Acknowledgements
- Persian NLP: `hazm` (`https://github.com/roshan-research/hazm`)
- Machine Learning: scikit‑learn, XGBoost
- Embeddings: Gensim Word2Vec

