# Machine Learning Practice Repository

A comprehensive collection of Jupyter notebooks demonstrating fundamental and advanced machine learning concepts, data preprocessing techniques, and predictive modeling using Python's data science ecosystem. This repository serves as a practical guide for learning and implementing ML workflows on real-world datasets.

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
│   ├── loan_data.csv          # Loan approval dataset
│   ├── cgpa_package.csv       # CGPA vs Package correlation
│   ├── house_price.csv        # House price prediction data
│   └── diabetes.csv           # Diabetes prediction dataset
├── notebooks/
│   ├── practice1.ipynb        # Data preprocessing fundamentals
│   ├── practice2.ipynb        # Feature scaling and transformations
│   ├── practice3.ipynb        # Feature selection techniques
│   ├── practice4.ipynb        # Train-test split and basic modeling
│   └── package_predictor.ipynb # Linear regression implementation
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── LICENSE                    # MIT License
└── .gitignore                 # Git ignore rules
```

## 📚 Notebooks Overview

### `notebooks/practice1.ipynb` - Data Preprocessing Fundamentals
- Loading and exploring datasets
- Handling missing values (mean/mode imputation)
- Categorical encoding (One-Hot, Label, Ordinal)
- Outlier detection (IQR method, Z-score)
- Data visualization and cleaning

### `notebooks/practice2.ipynb` - Feature Scaling & Transformations
- Standardization (StandardScaler)
- Normalization (MinMaxScaler)
- Log transformations for skewed data
- Handling duplicates and data types
- Advanced preprocessing techniques

### `notebooks/practice3.ipynb` - Feature Selection
- Sequential Feature Selection
- Forward and backward selection
- Model-based feature importance
- Dimensionality reduction concepts

### `notebooks/practice4.ipynb` - Basic Modeling Setup
- Train-test split implementation
- Data preparation for modeling
- Introduction to predictive workflows

### `notebooks/package_predictor.ipynb` - Linear Regression Project
- CGPA to Package prediction
- Model training and evaluation
- Visualization of regression results
- Prediction on new data

## 📊 Datasets Description

### Loan Data (`loan_data.csv`)
Financial dataset for loan approval prediction containing:
- Demographic information (age, gender, marital status)
- Financial metrics (income, credit score, loan amount)
- Property and employment details

### CGPA-Package Data (`cgpa_package.csv`)
Academic performance correlation dataset with:
- CGPA scores
- Corresponding job package offers
- Linear relationship analysis

### House Price Data (`house_price.csv`)
Real estate dataset including:
- Property features (area, bedrooms, bathrooms)
- Location factors (distance to city, crime rate)
- Market indicators (school rating, market trend)

### Diabetes Data (`diabetes.csv`)
Medical dataset for diabetes prediction with:
- Health metrics (glucose, blood pressure, BMI)
- Demographic factors (age, pregnancies)
- Binary classification target

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
