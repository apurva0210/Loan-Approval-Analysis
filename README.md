# Loan Approval Analysis

## 📌 Project Overview

Loan approval prediction is an important application of data analysis and machine learning in the financial industry. Banks and financial institutions evaluate applicant information such as income, CIBIL score, education, employment status, loan amount, repayment term, and asset values before making loan approval decisions.

This project focuses on analyzing loan applicant data using Python, performing exploratory data analysis (EDA), and applying multiple machine learning classification algorithms to predict loan approval status.

The project compares six classification algorithms using training accuracy, testing accuracy, and additional performance metrics to understand their predictive performance on the dataset.

## 🎯 Objectives

- Understand the structure and characteristics of the loan approval dataset.
- Perform data cleaning and preprocessing.
- Conduct exploratory data analysis to identify patterns and relationships.
- Analyze financial and applicant attributes associated with loan approval.
- Prepare features for machine learning classification.
- Apply multiple classification algorithms to predict loan approval status.
- Compare model performance using training and testing accuracy.
- Evaluate model predictions using confusion matrices and performance metrics.
- Identify important factors associated with loan approval decisions.

## 📂 Repository Contents

| File | Description |
|---|---|
| `Loan Approval Analysis.ipynb` | Jupyter Notebook containing data preprocessing, exploratory data analysis, and machine learning implementation. |
| `Loan Approval Analysis.pptx` | PowerPoint presentation summarizing the project, analysis, and findings. |
| `Loan Approval Analysis.pdf` | PDF version converted from the PowerPoint presentation. |
| `README.md` | Project documentation describing the objectives, dataset, methodology, and results. |

## 🛠️ Technologies and Libraries

- **Python** – Programming and model development.
- **Pandas** – Data manipulation and preprocessing.
- **NumPy** – Numerical computations.
- **Matplotlib** – Data visualization.
- **Seaborn** – Statistical data visualization.
- **Scikit-learn** – Machine learning algorithms and evaluation metrics.
- **Jupyter Notebook** – Interactive environment for analysis and experimentation.

## 🔍 Analysis Performed

### 1. Data Inspection

- Loaded and inspected the loan approval dataset.
- Examined the dataset structure, columns, and data types.
- Checked for missing values, duplicate records, and inconsistent entries.
- Identified the target variable used for classification.

### 2. Data Cleaning and Preprocessing

- Examined missing and inconsistent values.
- Reviewed numerical and categorical attributes.
- Prepared features for exploratory analysis and model training.
- Processed categorical and numerical variables as required.
- Cleaned column names where necessary.

### 3. Univariate Analysis

- Examined the distribution of annual income and loan amounts.
- Analyzed CIBIL scores and loan terms.
- Studied the distribution of loan approval status.
- Explored education, self-employment status, and number of dependents.
- Used visualizations to understand individual feature distributions.

### 4. Bivariate Analysis

- Examined the relationship between CIBIL score and loan approval.
- Explored annual income in relation to approval status.
- Investigated loan amount and repayment term patterns.
- Analyzed residential, commercial, luxury, and bank asset values.
- Compared applicant characteristics across loan approval categories.

### 5. Feature Analysis

The project examines the following applicant attributes:

- **CIBIL Score:** Represents the applicant's creditworthiness.
- **Annual Income:** Provides information about earning capacity.
- **Loan Amount:** Represents the requested financing.
- **Loan Term:** Indicates the repayment duration.
- **Education:** Describes the applicant's education status.
- **Self-Employment Status:** Indicates whether the applicant is self-employed.
- **Number of Dependents:** Represents the number of people financially dependent on the applicant.
- **Asset Values:** Includes residential, commercial, luxury, and bank assets.

### 6. Machine Learning Classification

The following six classification algorithms were applied to predict loan approval status:

1. Logistic Regression
2. Support Vector Machine (SVM)
3. Decision Tree Classifier
4. Random Forest Classifier
5. Naive Bayes
6. K-Nearest Neighbors (KNN)

The models were compared using training accuracy, testing accuracy, confusion matrices, and the additional metrics calculated in the notebook.

## 📊 Model Performance and Evaluation

### Training and Testing Accuracy

The following table presents the accuracy values obtained from your notebook. Percentages are calculated from the decimal scores you provided.

| Algorithm | Training Accuracy | Testing Accuracy |
|---|---:|---:|
| Logistic Regression | 62.99% | 60.58% |
| Support Vector Machine (SVM) | 62.95% | 60.50% |
| Decision Tree Classifier | 96.39% | 94.61% |
| Random Forest Classifier | 100.00% | 96.25% |
| Naive Bayes | 93.24% | 91.96% |
| K-Nearest Neighbors (KNN) | 94.65% | 90.55% |

### Key Observations

- **Random Forest Classifier:** Achieved the highest testing accuracy of **96.25%**. Its training accuracy was 100%, indicating a possible overfitting risk that should be investigated.
- **Decision Tree Classifier:** Achieved 94.61% testing accuracy and 96.39% training accuracy, showing strong classification performance with a smaller training–testing accuracy gap than Random Forest.
- **Naive Bayes:** Achieved 91.96% testing accuracy, providing good predictive performance among the evaluated models.
- **K-Nearest Neighbors:** Achieved 90.55% testing accuracy, although its test performance was lower than its training accuracy.
- **Logistic Regression:** Achieved 60.58% testing accuracy.
- **Support Vector Machine:** Achieved 60.50% testing accuracy, similar to Logistic Regression.

Based on testing accuracy, Random Forest performed best, followed by Decision Tree, Naive Bayes, and KNN. Logistic Regression and SVM had comparatively lower testing accuracy.

### Confusion Matrix Analysis

Confusion matrices were used to examine the distribution of correct and incorrect predictions for the classification models.

According to the notebook's reported analysis:

- Logistic Regression and SVM showed comparatively more misclassifications.
- Decision Tree and Random Forest showed fewer combined false-positive and false-negative predictions.
- Naive Bayes and KNN demonstrated intermediate classification performance.

The reported combined false-positive and false-negative counts were 505 for Logistic Regression and SVM, 49 for Decision Tree and Random Forest, 103 for Naive Bayes, and 121 for KNN.

These counts should be interpreted with the underlying confusion matrices because false positives and false negatives represent different types of classification errors.

### Additional Performance Metrics

The notebook also reports R² score, Mean Squared Error (MSE), and Mean Absolute Error (MAE).

| Algorithm | Reported R² Score | Reported MSE | Reported MAE |
|---|---:|---:|---:|
| Logistic Regression | -0.6496 | 0.3942 | 0.3942 |
| SVM | -0.6529 | 0.3950 | 0.3950 |
| Decision Tree | 0.8432 | 0.0375 | 0.0375 |
| Random Forest | 0.8432 | 0.0375 | 0.0375 |
| Naive Bayes | 0.6635 | 0.0804 | 0.0804 |
| KNN | 0.6047 | 0.0945 | 0.0945 |

**Metric interpretation:** Since loan approval is a classification task, accuracy, precision, recall, F1-score, and confusion matrices are generally more appropriate than R², MSE, and MAE. The latter metrics should only be interpreted if they were calculated on a meaningful numerical representation of the target. The identical Decision Tree and Random Forest values should also be verified against the original notebook outputs.

## 📈 Key Findings from Exploratory Data Analysis

The exploratory analysis identified several patterns associated with loan approval:

1. **CIBIL Score:** Applicants with higher CIBIL scores showed a greater likelihood of loan approval.
2. **Number of Dependents:** The analysis suggested that a higher number of dependents was associated with a lower likelihood of approval.
3. **Asset Ownership:** Higher residential, commercial, luxury, and bank asset values were associated with loan approval patterns.
4. **Loan Amount and Tenure:** The analysis explored the relationship between loan amounts, repayment duration, and approval outcomes.

These are observed associations in the dataset and do not establish causation or universal lending rules.

## 💡 Business Applications

Loan approval analysis demonstrates how data analysis and machine learning can support the understanding of financial applicant profiles.

Potential applications include:

- Analyzing applicant financial profiles.
- Identifying factors associated with loan approval.
- Supporting preliminary credit risk assessment.
- Comparing machine learning algorithms for financial classification.
- Understanding data-driven loan evaluation workflows.
- Exploring applicant characteristics for further analysis.

This project is intended for educational purposes. Real-world lending decisions require additional validation, fairness assessment, regulatory compliance, and appropriate human oversight.

## ▶️ How to Run

### Step 1: Clone the Repository

```bash
git clone https://github.com/apurva0210/Loan-Approval-Analysis.git
```

### Step 2: Navigate to the Project Directory

```bash
cd Loan-Approval-Analysis
```

### Step 3: Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Step 4: Download the Dataset

Download the dataset from Kaggle using the link in the Dataset section.

### Step 5: Configure the Dataset Path

Place the downloaded CSV file in your local project directory and update the file path in the notebook.

### Step 6: Run Jupyter Notebook

```bash
jupyter notebook
```

Open `Loan Approval Analysis.ipynb` and execute the notebook cells sequentially.

## 📁 Dataset

The dataset used in this project is available on Kaggle.

**Dataset:** [Loan Approval Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)

The dataset contains 4,269 records and 13 columns describing applicants' financial information, credit scores, and assets.

### Dataset Features

| Column | Description |
|---|---|
| `loan_id` | Unique loan application identifier |
| `no_of_dependents` | Number of dependents |
| `education` | Applicant's education status |
| `self_employed` | Whether the applicant is self-employed |
| `income_annum` | Applicant's annual income |
| `loan_amount` | Requested loan amount |
| `loan_term` | Loan repayment term |
| `cibil_score` | Applicant's credit score |
| `residential_assets_value` | Value of residential assets |
| `commercial_assets_value` | Value of commercial assets |
| `luxury_assets_value` | Value of luxury assets |
| `bank_asset_value` | Value of bank assets |
| `loan_status` | Target variable indicating loan approval status |

**Target Variable:** `loan_status`

The dataset is not included in this repository. Download it separately from Kaggle and update the dataset path in the notebook before running the analysis.

## 📑 Project Presentation

The project presentation is available in PowerPoint and PDF formats:

- [View PowerPoint Presentation](https://github.com/apurva0210/Loan-Approval-Analysis/blob/main/Loan%20Approval%20Analysis.pptx)
- [View PDF Presentation](https://github.com/apurva0210/Loan-Approval-Analysis/blob/main/Loan%20Approval%20Analysis.pdf)

The PDF is a converted version of the PowerPoint presentation, provided for convenient viewing and sharing.

## 👩‍💻 Author

**Apurva Kadam**

M.Sc. Information Technology

GitHub: [apurva0210](https://github.com/apurva0210)

## 📝 Conclusion

This project demonstrates the use of exploratory data analysis and machine learning classification to predict loan approval status. It examines applicant financial information, CIBIL scores, loan characteristics, and asset values to understand their relationships with approval outcomes.

Among the six evaluated models, Random Forest achieved the highest testing accuracy of **96.25%**, followed by Decision Tree at **94.61%**. Naive Bayes achieved 91.96%, while KNN achieved 90.55%. Logistic Regression and SVM produced lower testing accuracies of approximately 60.5%.

The results demonstrate the importance of comparing multiple classification algorithms using held-out test data. Further validation, error analysis, and hyperparameter tuning could help assess the models' generalization and reliability.

---

*This project was developed for learning and practical application of data analysis and machine learning concepts.*
