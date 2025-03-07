# House Price Prediction Using Advanced Regression Techniques

## Overview
This project tackles the Kaggle House Prices - Advanced Regression Techniques competition. The goal is to predict residential home prices in Ames, Iowa, using advanced regression models and feature engineering techniques. The solution involves data preprocessing, exploratory data analysis (EDA), feature engineering, and ensemble modeling to achieve competitive prediction accuracy.

## Dataset
The dataset is provided by Kaggle and contains 79 explanatory variables describing various aspects of residential homes. It includes both numerical and categorical features.
- **Train Data**: `train.csv` (1,460 entries)
- **Test Data**: `test.csv` (1,459 entries)
- **Dataset Obtained From Competition**: [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

## Key Steps

### 1. Data Preprocessing
- **Handling Missing Values**:  
  - Filled missing categorical features (e.g., `Alley`, `FireplaceQu`) with "No".
  - Filled missing numerical features (e.g., `LotFrontage`, `MasVnrArea`) with 0 or mean values.
- **Outlier Removal**: Identified and removed outliers using scatter plots and Z-scores for features like `LotFrontage`, `GrLivArea`, and `TotalBsmtSF`.

### 2. Feature Engineering
- **New Features**:  
  - `houseage`: Years since the house was built.
  - `totalsf`: Total square footage (1st floor + 2nd floor + basement areas).
  - `totalbaths`: Sum of full and half bathrooms.
  - `totalporchsf`: Combined porch areas.
- **Dropped Redundant Features**: Removed highly correlated features (e.g., `GarageArea` vs. `GarageCars`).

### 3. Model Training
- **Algorithms Tested**:  
  - Linear Regression
  - Random Forest Regressor
  - XGBoost
  - Ridge Regression
  - Gradient Boosting Regressor
  - LightGBM
  - CatBoost
- **Hyperparameter Tuning**: Used `GridSearchCV` for optimizing model parameters.
- **Ensemble Methods**:  
  - **Voting Regressor**: Combined Gradient Boosting, XGBoost, and Ridge Regression.
  - **Stacking Regressor**: Stacked top-performing models with a meta-learner.

### 4. Results
- **Best Model**: Stacking Regressor achieved the lowest RMSE.
- **Key Metrics**:  
  | Model              | RMSE (Validation) |
  |---------------------|--------------------|
  | Linear Regression   | 0.144             |
  | Random Forest       | 0.132             |
  | XGBoost             | 0.123             |
  | Stacking Regressor  | **0.118**         |

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/housing-prices-prediction.git
   ```

2. **Install dependencies:**
   ```bash
   pip install pandas numpy scipy scikit-learn matplotlib seaborn xgboost catboost lightgbm streamlit
   ```

## Usage

1. **Download the dataset**  place `train.csv` and `test.csv` in the project directory.

2. **Run the Jupyter Notebook** `Housing-Data-Model.ipynb` step by step:
   - Data loading and inspection
   - Exploratory Data Analysis (EDA)
   - Data preprocessing
   - Model training and evaluation
   - Results visualization

## Key Visualizations
- `SalePrice` vs `MSSubClass`: Example visualization from EDA showing feature relationships.

## Dependencies
- Python 3.7+
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `xgboost`
- `catboost`
- `lightgbm`

### Future Improvements:
- Advanced feature engineering
- Hyperparameter tuning
- Ensembling optimization
- Detailed error analysis
- use of Neural Networks
