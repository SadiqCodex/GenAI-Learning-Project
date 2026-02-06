# Machine Learning Practice Repository

A comprehensive collection of Jupyter notebooks demonstrating fundamental and advanced machine learning concepts, data preprocessing techniques, and predictive modeling using Python's data science ecosystem. This repository serves as a practical guide for learning and implementing ML workflows on real-world datasets, covering supervised and unsupervised learning, ensemble methods, and model evaluation.

## Table of Contents
- [Features](#-features)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Notebooks Overview](#-notebooks-overview)
- [Datasets Description](#-datasets-description)
- [Usage Examples](#-usage-examples)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)
- [Contact](#-contact)
- [Acknowledgments](#-acknowledgments)

## 🚀 Features

- **Data Preprocessing**: Complete pipelines for handling missing values, outliers, and data transformations
- **Feature Engineering**: Encoding categorical variables, feature scaling, and selection techniques
- **Machine Learning Models**: Implementation of regression, classification, and predictive modeling
- **Data Visualization**: Exploratory data analysis with matplotlib and seaborn
- **Real Datasets**: Practice on diverse datasets including loan data, CGPA-package correlation, house prices, and diabetes prediction
- **Best Practices**: Clean, documented code following Python and ML conventions

## 📋 Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab
- Required packages: `pip install -r requirements.txt`

## 🛠 Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd machine-learning-practice
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv ml_env
   source ml_env/bin/activate  # On Windows: ml_env\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

## 📊 Project Structure

```
machine-learning-practice/
├── datasets/
│   ├── cgpa_package.csv          # CGPA vs Package correlation
│   ├── cgpa_score_placement.csv  # CGPA, score, and placement data
│   ├── citizen_data.csv          # Citizen demographic data
│   ├── diabetes.csv              # Diabetes prediction dataset
│   ├── grocery.csv               # Grocery transaction data for association rules
│   ├── house_price.csv           # House price prediction data
│   ├── iris_data.csv             # Iris flower dataset (features only)
│   ├── iris_data_species.csv     # Iris flower dataset with species labels
│   ├── iris_multiclass.csv       # Multiclass iris data
│   ├── level_salary.csv          # Experience level vs salary data
│   ├── loan_data.csv             # Loan approval dataset
│   ├── logistic_dataset.csv      # Binary classification dataset
│   ├── regularization_house_dataset.csv # House data for regularization
│   ├── salary_dataset.csv        # Salary prediction dataset
│   ├── salary_new_dataset.csv    # Additional salary data
│   ├── student_data.csv          # Student performance data
│   ├── student.csv               # Student information
│   └── subscription_dataset.csv  # Subscription prediction data
├── notebooks/
│   ├── practice1.ipynb           # Data preprocessing fundamentals
│   ├── practice2.ipynb           # Feature scaling and transformations
│   ├── practice3.ipynb           # Feature selection techniques
│   ├── practice4.ipynb           # Train-test split and basic modeling
│   ├── package_predictor.ipynb   # Linear regression implementation
│   ├── multiple_linear.reg.ipynb # Multiple linear regression
│   ├── polynomial_classifier.ipynb # Polynomial classification
│   ├── polynomial_salary_model.ipynb # Polynomial regression for salary
│   ├── regression_predictor.ipynb # Regression modeling
│   ├── decision_tree.ipynb       # Decision tree classification
│   ├── multiple_decision_tree.ipynb # Multiple decision trees
│   ├── k_nearest-classificatoin.ipynb # KNN classification
│   ├── k_nearest-regrassor.ipynb # KNN regression
│   ├── Support Vector Machines(SVM) - Classification.ipynb # SVM classification
│   ├── multiple_classification.ipynb # Multiple classification algorithms
│   ├── subscription_prediction.ipynb # Subscription prediction project
│   ├── job_predictino.ipynb      # Job prediction (typo in filename)
│   ├── imbalance_dataset.ipynb   # Handling imbalanced datasets
│   ├── imblearn.ipynb           # Imbalanced learning techniques
│   ├── confusion_matrix.ipynb    # Confusion matrix and metrics
│   ├── Bias–Variance.ipynb       # Bias-variance tradeoff
│   ├── dataset_preprocessing.ipynb # Advanced preprocessing
│   ├── bayes_theroam_dataset.ipynb # Bayesian theorem applications
│   ├── esemble_learning.ipynb    # Ensemble learning methods
│   └── unsupervised_k_mean_cluster.ipynb # Unsupervised learning (K-means, clustering)
├── Matplotlib/                   # Matplotlib visualization examples
├── Numpy/                        # NumPy practice notebooks
├── Pandas/                       # Pandas data manipulation examples
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── LICENSE                       # MIT License
└── .gitignore                    # Git ignore rules
```

## 📚 Notebooks Overview

### Data Preprocessing & Feature Engineering
- **`practice1.ipynb`** - Data preprocessing fundamentals: missing values, encoding, outliers, visualization
- **`practice2.ipynb`** - Feature scaling and transformations: standardization, normalization, log transforms
- **`practice3.ipynb`** - Feature selection techniques: sequential selection, importance ranking
- **`dataset_preprocessing.ipynb`** - Advanced preprocessing pipelines and techniques

### Regression Models
- **`package_predictor.ipynb`** - Linear regression: CGPA to package prediction with visualization
- **`multiple_linear.reg.ipynb`** - Multiple linear regression: multi-feature salary prediction
- **`polynomial_salary_model.ipynb`** - Polynomial regression: non-linear salary modeling
- **`regression_predictor.ipynb`** - General regression modeling and evaluation

### Classification Models
- **`decision_tree.ipynb`** - Decision tree classification: subscription prediction with visualization
- **`multiple_decision_tree.ipynb`** - Advanced decision tree techniques and hyperparameter tuning
- **`k_nearest-classificatoin.ipynb`** - KNN classification: neighbor-based prediction algorithms
- **`k_nearest-regrassor.ipynb`** - KNN regression: distance-based continuous prediction
- **`Support Vector Machines(SVM) - Classification.ipynb`** - SVM classification: kernel methods and hyperparameter optimization
- **`multiple_classification.ipynb`** - Comparative analysis of multiple classification algorithms
- **`polynomial_classifier.ipynb`** - Polynomial classification: non-linear decision boundaries

### Ensemble Learning
- **`esemble_learning.ipynb`** - Ensemble methods: bagging, boosting, voting classifiers/regressors
- **`Bias–Variance.ipynb`** - Bias-variance tradeoff analysis in ensemble contexts

### Specialized Topics
- **`imbalance_dataset.ipynb`** - Handling imbalanced datasets: techniques for skewed class distributions
- **`imblearn.ipynb`** - Advanced imbalanced learning methods and evaluation metrics
- **`confusion_matrix.ipynb`** - Model evaluation: confusion matrices, precision, recall, F1-score
- **`bayes_theroam_dataset.ipynb`** - Bayesian theorem applications in machine learning
- **`job_predictino.ipynb`** - Job prediction modeling (note: filename has typo)
- **`subscription_prediction.ipynb`** - End-to-end subscription prediction project

### Unsupervised Learning
- **`unsupervised_k_mean_cluster.ipynb`** - K-means clustering, hierarchical clustering, DBSCAN, association rules

### Model Validation & Cross-Validation
- **`practice4.ipynb`** - Basic modeling setup: train-test splits, cross-validation fundamentals

## 📊 Datasets Description

### Core Datasets
- **`loan_data.csv`** - Loan approval prediction: Demographic and financial data for credit risk assessment
- **`cgpa_package.csv`** - Academic performance: CGPA scores correlated with job package offers
- **`cgpa_score_placement.csv`** - Placement data: CGPA, test scores, and placement outcomes
- **`house_price.csv`** - Real estate: Property features, location factors, and market indicators
- **`diabetes.csv`** - Medical prediction: Health metrics for diabetes classification
- **`subscription_dataset.csv`** - Marketing: Customer data for subscription prediction
- **`salary_dataset.csv`** - Compensation: Age, experience, and salary relationships

### Specialized Datasets
- **`iris_data.csv`** / **`iris_data_species.csv`** - Classic ML dataset: Flower measurements and species classification
- **`logistic_dataset.csv`** - Binary classification: Synthetic data for logistic regression
- **`level_salary.csv`** - Experience-salary: Position levels mapped to compensation
- **`grocery.csv`** - Market basket: Transaction data for association rule mining
- **`regularization_house_dataset.csv`** - Regularization practice: House price data with multicollinearity

### Additional Datasets
- **`citizen_data.csv`** - Demographic data for citizen analysis
- **`iris_multiclass.csv`** - Multiclass classification: Extended iris dataset
- **`salary_new_dataset.csv`** - Extended salary data
- **`student_data.csv`** / **`student.csv`** - Academic performance datasets

## 🏃‍♂️ Usage Examples

### Running Individual Notebooks
```bash
jupyter notebook notebooks/practice1.ipynb
```

### Basic Data Loading and Preprocessing
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv('datasets/loan_data.csv')

# Handle missing values
data.fillna(data.mean(), inplace=True)

# Scale numerical features
scaler = StandardScaler()
data['scaled_income'] = scaler.fit_transform(data[['Annual_Income']])
```

### Linear Regression Example
```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load CGPA-Package data
cgpa_data = pd.read_csv('datasets/cgpa_package.csv')
X = cgpa_data[['cgpa']]
y = cgpa_data['package']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Make predictions
predictions = lr.predict(X_test)
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to functions
- Include comments for complex logic
- Test your code thoroughly
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Sadik Mohammad**

## 📞 Contact

- **Author**: Sadik Mohammad
- **GitHub**: [Your GitHub Profile]
- **Email**: [Your Email]

For questions, suggestions, or collaboration opportunities, please open an issue in the repository.

## 🙏 Acknowledgments

- Thanks to the open-source community for providing excellent libraries
- Dataset sources: Various public repositories and Kaggle
- Inspired by real-world machine learning workflows

---

*Built with ❤️ using Python, pandas, scikit-learn, matplotlib, seaborn, and Jupyter*
