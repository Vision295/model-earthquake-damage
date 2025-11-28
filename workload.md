# ✅ **Balanced Workload Split (3 Equal Tracks)**

## **Member 1 — Model Selection & Hyperparameter Work (Independent Track)**

**Focus:** Experimentation with models and tuning
**Main tasks:**

1. Choose **2–3 candidate AI models**
   Examples depending on task:

   * Classification → Random Forest, XGBoost, Logistic Regression, SVM
   * Regression → Linear Regression, Random Forest, XGBoost
   * Deep learning → MLP, CNN, LSTM (depending on data type)

2. Decide initial **hyperparameter grids**

   * Learning rates
   * Tree depth / number of estimators
   * Regularization strengths
   * Batch sizes, epochs, activation functions, etc.

3. Perform **hyperparameter tuning**

   * Grid search
   * Random search
   * Bayesian optimization (optional)

4. Compare models based on:

   * Validation accuracy / RMSE / F1-score
   * Training time
   * Generalization performance

5. Produce the **final model selection report**

**Outcome:** Most promising model + tuned hyperparameters.

## **Member 2 – Preprocessing Track A: Data Understanding + EDA + Documentation**

This track is now **lighter**, focusing on analysis rather than heavy feature-selection algorithms.

### **Main Tasks**

1. **Exploratory Data Analysis (EDA)**

   * Summary statistics
   * Class/target distribution
   * Correlation matrix
   * Outlier exploration
   * Distribution plots

2. **Data Documentation**

   * Describe data types
   * Meaning of variables
   * Initial insights and patterns
   * Potential transformations (suggestions only)

3. **Leakage check**

   * Ensure no columns leak target information

4. **Train/Validation/Test Split Setup**

   * Choose the splitting strategy
   * If classification → use stratified splitting
   * Provide these splits to Track C and Member 1

**Workload Level:** Moderate
**Outcome:** Clean documentation + EDA + final data splits
(This complements Track C without overlapping too much.)

---

## **Member 3 – Preprocessing Track B: Feature Engineering + Feature Selection + Data Transformations**

This track is now reduced but still technically focused—balancing the effort with Track A.

### **Main Tasks**

1. **Feature Engineering (light but meaningful)**

   * One-hot encoding / label encoding
   * Scaling (StandardScaler/MinMax)
   * Optional: 1–2 domain-inspired new features

2. **Feature Selection (but limited scope)**
   Implement **only 1–2 strong feature selection methods**, such as:

   * Mutual information
   * Tree-based feature importance
   * ANOVA F-test
     (No heavy recursive methods like RFE unless all agree.)

3. **Feature transformation**

   * Log transform skewed features
   * Normalization if model-dependent
   * Remove constant/low-variance features

4. Package everything into a **Sklearn Pipeline**

**Workload Level:** Moderate
**Outcome:** A ready-to-use preprocessing pipeline + selected features

---

# 🎯 **Balanced Overview Table**

| Member | Task Category                                     | Why It’s Balanced                                          |
| ------ | ------------------------------------------------- | ---------------------------------------------------------- |
| **1**  | Model selection & hyperparameter tuning           | Full independent track—high experimentation effort         |
| **2**  | EDA + documentation + splitting                   | Analysis-focused, similar in weight to light engineering   |
| **3**  | Feature engineering + selection + transformations | Technical preprocessing but reduced to moderate complexity |
