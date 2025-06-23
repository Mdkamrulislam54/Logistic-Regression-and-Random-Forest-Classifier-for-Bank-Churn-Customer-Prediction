# 🤖 Bank Churn Prediction — Logistic Regression & Random Forest ✨
#🎯 Objective
Develop and compare two machine learning models—Logistic Regression and Random Forest—to predict whether bank customers are likely to churn, enabling proactive retention strategies.

# 🔍 Business Context
Reducing customer churn is vital for banking profitability. Predicting churn enables targeted interventions like personalized offers or retention campaigns to hold onto at-risk customers.

# 🧰 Dataset & Preprocessing
Data: Includes features such as CreditScore, Age, Balance, NumOfProducts, IsActiveMember, EstimatedSalary, Geography, Gender, and the churn flag Exited.

Cleaning Steps: Managed missing values, encoded categorical variables via One‑Hot Encoding, and scaled numerical features.

# ⚙️ Modeling Pipeline
**Train-Test Split**
Split data into training and testing sets (e.g., 80/20 stratified split) to preserve class balance.

**Modeling Approaches**

Logistic Regression: Evaluates baseline performance on probability-based classification.

Random Forest Classifier: An ensemble using multiple decision trees with bagging.

**Hyperparameter Tuning**

Random Forest: Tweaked n_estimators, max_depth, and max_leaf_nodes.

Logistic Regression: Explored variations in fit_intercept.

**Evaluation Metrics**
Used accuracy, precision, recall, F1-score, and confusion matrix.

# 📈 Model Performance & Comparison

**Initial Run**: Random Forest outperformed Logistic Regression (e.g., Random Forest ~86% accuracy vs. Logistic Regression ~83%) 
**Feature Impact**: Removing less predictive features slightly reduced accuracy for both models, indicating limited redundancy
**Final Models:**

Random Forest: ~86% overall accuracy, better stability and feature resilience.

Logistic Regression: ~83% accuracy, with marginal decline after feature removal.

# 🌟 Key Insights

**Random Forest** offers higher accuracy and robustness.

The Recall metric is critical to ensure high-risk customers are not missed.

Careful feature selection and tuning improve both model performance and interpretability.

# ✅ Why It Matters

Empowers banks to predict and reduce customer churn with data-driven decisions.

Random Forest’s superior stability makes it ideal for deployment.

Code is modular and extensible for other banking datasets or deployment pipelines.
