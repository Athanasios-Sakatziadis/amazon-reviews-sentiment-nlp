


# Sentiment Classification of Amazon Mobile Reviews: Logistic Regression vs. XGBoost


An end-to-end Natural Language Processing (NLP) and Machine Learning project designed to classify customer sentiment from over 400,000 Amazon unlocked mobile phone reviews. The workflow emphasizes rigorous data cleaning to prevent data leakage, TF-IDF feature extraction with n-grams, a performance showdown between linear and tree-based classifiers, and model explainability.

> **Academic Context:** This project was developed as an optimization assignment for the **Web and Text Analytics** course at the **University of Macedonia** (supervised by **Prof. Evangelos Kalampokis**). The objective was to take the baseline `.ipynb` solution presented in class and significantly improve its methodology, robustness, and analytical depth.

**Dataset Source:** [Amazon Unlocked Mobile Reviews (Kaggle)](https://www.kaggle.com/datasets/nehasontakke/amazon-unlocked-mobilecsv)

---

## Project Overview


In e-commerce, automatically understanding customer feedback at scale is critical for product improvement and customer retention. This project transforms raw, unstructured customer reviews into actionable insights by framing the problem as a binary classification task:
* **Positive Sentiment (`1`):** Ratings of 4 and 5 stars.
* **Negative Sentiment (`0`):** Ratings of 1 and 2 stars.
*(Neutral 3-star reviews were excluded to eliminate sentiment ambiguity).*

---

## Key Improvements Over the Class Baseline


* **Data Leakage Prevention:** Identified and removed over 60% duplicate `(review, rating)` pairs *prior* to the train-test split to ensure unbiased evaluation and prevent artificially inflated metrics.
* **Algorithmic Showdown (Linear vs. Tree-Based):** Expanded beyond the baseline Logistic Regression model by introducing an **XGBoost Classifier** to benchmark how gradient-boosted trees perform against linear models on high-dimensional sparse text data.
* **Advanced Evaluation (ROC-AUC):** Supplemented standard accuracy metrics with comparative **ROC Curves** and **AUC scores** for a deeper assessment of class separation.
* **Model Explainability:** Added feature weight extraction and visualization to interpret the top positive and negative n-grams driving the model's predictions.

---

## Methodology & Pipeline


1. **Data Cleaning & Leakage Prevention:**
   * Handled missing values in the primary text feature (`reviews`).
   * **Deduplication:** Discovered that over 60% of the raw dataset (413,840 rows) consisted of duplicate `(review, rating)` pairs. Removing these prior to splitting prevented **data leakage** between the training and test sets and eliminated frequency bias in text vectorization.


2. **Feature Engineering (TF-IDF):**
   * Converted raw text into numerical sparse matrices using `TfidfVectorizer`.
   * Configured `min_df=5` to filter out rare noise/typos and `ngram_range=(1, 2)` to capture both single words (**unigrams**) and two-word context (**bigrams**, e.g., *"not happy"*, *"no problems"*).
   * Fitted the vocabulary **exclusively on the training set** (75/25 split) before transforming the test set.


3. **Model Showdown (Linear vs. Tree-Based):**
   * Trained and compared **Logistic Regression** (well-suited for high-dimensional sparse text matrices) against **XGBoost** (gradient boosted decision trees).


4. **Model Explainability:**
   * Extracted and visualized the top positive and negative model coefficients to interpret which specific words and phrases drive customer satisfaction or dissatisfaction.

---

## Model Evaluation & Results


Both models were evaluated on the unseen test set using Precision, Recall, F1-Score, and the Area Under the ROC Curve (ROC-AUC).


| Model | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score | ROC-AUC Score | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | `[e.g., 95%]` | `[e.g., 0.96]` | `[e.g., 0.97]` | `[e.g., 0.97]` | **`[e.g., 0.9850]`** | `[e.g., 4.2s]` |
| **XGBoost Classifier** | `[e.g., 93%]` | `[e.g., 0.94]` | `[e.g., 0.96]` | `[e.g., 0.95]` | `[e.g., 0.9720]` | `[e.g., 45.0s]` |

### Key Takeaways


* **Logistic Regression Outperforms on Sparse Text:** Logistic Regression achieved a higher ROC-AUC score while training significantly faster than XGBoost, demonstrating why linear models remain a strong benchmark for high-dimensional bag-of-words/TF-IDF representations.


* **Context Matters (Bigrams):** Feature importance analysis confirmed that capturing bigrams allowed the model to accurately weigh negation phrases like *"not happy"* (strong negative driver) and *"no problems"* (strong positive driver), alongside domain-specific e-commerce terms like *"return"*.

---

## Screenshots & Visual Results

### 1. Target Class Distribution (Exploratory Data Analysis)
*Distribution of binary sentiment labels after deduplication and removal of neutral 3-star reviews (~105k positive vs. ~45k negative).*


<img width="569" height="391" alt="Screenshot 2026-11-15 182054" src="https://github.com/user-attachments/assets/0d2f0bc7-c309-4881-b72d-08c725817133" />


### 2. Model Training Execution Time (Computational Efficiency)
*Benchmarking training speed on high-dimensional TF-IDF matrices: Logistic Regression completed in **1.16s** compared to **80.25s** for XGBoost.*


<img width="1143" height="128" alt="Screenshot 2026-11-15 182245" src="https://github.com/user-attachments/assets/ab103653-b40b-410e-9cf6-8c751cdb8ce8" />




### 3. ROC Curve Comparison (Logistic Regression vs. XGBoost)
*Visualizing classification performance across thresholds: Logistic Regression achieved an AUC of **0.98**, slightly outperforming XGBoost (**0.97**).*


<img width="680" height="547" alt="Screenshot 2026-11-15 182110" src="https://github.com/user-attachments/assets/77ab5d99-872b-4d55-974d-9a38489d5e96" />


### 4. Feature Importance & Model Explainability (Top 10 N-Grams)
*Extracting Logistic Regression coefficient weights to highlight the strongest positive and negative unigrams/bigrams (e.g., "no problems", "not happy", "return").*


<img width="1091" height="447" alt="Screenshot 2026-11-15 182126" src="https://github.com/user-attachments/assets/73b08085-d445-49f6-99e5-626ef3d933d7" />


