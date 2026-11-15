# Sentiment Classification of Amazon Mobile Reviews: Logistic Regression vs. XGBoost

## Project Overview
This project explores the application of Natural Language Processing (NLP) and Machine Learning techniques to sentiment classification using more than 400,000 Amazon mobile phone reviews. 

The objective is to classify customer reviews as either positive or negative, using review text and star ratings. The project compares two classification algorithms — **Logistic Regression** and **XGBoost** — and examines their predictive performance, computational efficiency, and interpretability.

> **Academic Context:** The work was developed as an optimization of a baseline solution presented in the **Web and Text Analytics** course at the **University of Macedonia**, under the supervision of **Professor Evangelos Kalampokis**. The original approach was extended through additional preprocessing, improved evaluation, model comparison, and feature analysis.

**Dataset:** The analysis uses the [Amazon Unlocked Mobile Reviews](https://www.kaggle.com/datasets/nehasontakke/amazon-unlocked-mobilecsv) dataset.

---

## Target Variable
The original five-star rating system was transformed into a binary sentiment classification problem:
* **Positive (1):** Ratings of 4 or 5 stars.
* **Negative (0):** Ratings of 1 or 2 stars.
*(Neutral 3-star reviews were excluded from the analysis).*

This approach focuses the classification task on clearly positive and negative customer opinions.

---

## Key Improvements Over Baseline
Compared with the baseline implementation, this project introduces several improvements:
* **Duplicate removal:** Duplicate `(review, rating)` pairs were removed before the train-test split, reducing the risk of duplicate observations appearing in both datasets and influencing evaluation results.
* **Expanded model comparison:** XGBoost was introduced alongside Logistic Regression to compare a linear model with a gradient-boosting approach.
* **Improved evaluation:** ROC curves and ROC-AUC scores were included to provide an additional perspective on classification performance.
* **Feature analysis:** Model coefficients were examined to identify words and phrases associated with positive and negative sentiment.
* **Computational efficiency:** Training times were recorded to compare the computational cost of the two approaches.

---

## Methodology

### 1. Data Cleaning and Preprocessing
More than 60% of the original records were identified as duplicate `(review, rating)` pairs. These duplicates were removed before splitting the data into training and testing sets. After cleaning and removing neutral ratings, the final dataset contained approximately 149,343 reviews (104,819 Positive, 44,524 Negative).

### 2. Train-Test Split
The cleaned dataset was divided into **75% training data** and **25% testing data**. The split was performed *after* duplicate removal to reduce the risk of overly optimistic evaluation caused by data leakage.

### 3. Text Vectorization (TF-IDF)
The review text was transformed into numerical features using `TfidfVectorizer` configured with:
* `min_df = 5`
* `ngram_range = (1, 2)` (capturing both **unigrams** and **bigrams** like *"not happy"*).
The TF-IDF vocabulary was fitted exclusively on the training data.

### 4. Model Training & Evaluation
Trained and compared **Logistic Regression** (linear benchmark) against **XGBoost** (gradient boosting). Models were evaluated using Accuracy, Precision, Recall, F1-score, ROC-AUC, and Training time.

---

## Results and Model Comparison

The results indicate that Logistic Regression achieved stronger overall performance in this experiment while requiring substantially less training time than XGBoost.

| Metric | Logistic Regression | XGBoost |
| :--- | :---: | :---: |
| **Accuracy** | ~94% | ~92% |
| **ROC-AUC Score** | **0.9793** | 0.9673 |
| **Negative-class F1-score** | ~0.89 | ~0.85 |
| **Positive-class F1-score** | ~0.96 | ~0.94 |
| **Training Time** | **1.16 sec** | 80.25 sec |

### Key Findings
* **Logistic Regression achieved the stronger results:** It obtained a higher ROC-AUC score and better classification performance across both sentiment classes.
* **Training efficiency:** Logistic Regression completed training in approximately 1.16 seconds, compared with 80.25 seconds for XGBoost.
* **Text representation matters:** Using unigrams and bigrams allows the model to represent multi-word expressions, including negation phrases, rather than relying exclusively on individual words.
* **Interpretable features:** The coefficient analysis highlights terms associated with positive and negative customer sentiment, including domain-specific expressions such as *"return"*.

---

## Screenshots & Visual Results

### 1. Target Class Distribution
![Target Class Distribution](images/class_distribution.png)

### 2. Model Training Execution Time
![Training Time](images/training_time.png)

### 3. ROC Curve Comparison 
![ROC Curve](images/roc_curve.png)

### 4. Feature Importance & Model Explainability
![Feature Importance](images/feature_importance.png)

---
