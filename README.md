# Practical 1: House Price Prediction Using Regression

## Overview
This practical explores supervised machine learning for predicting house prices using a real estate dataset containing 4,200 records. It compares Simple Linear Regression, Multiple Linear Regression, and degree-3 Polynomial Regression, and demonstrates gradient descent using `SGDRegressor`.

## Objectives
- Understand supervised learning and regression.
- Train a Simple Linear Regression model using house area.
- Train a Multiple Linear Regression model using the available features.
- Apply degree-3 Polynomial Features and compare model performance.
- Evaluate predictions using regression metrics.
- Explore gradient descent, bias, variance, overfitting, and underfitting.

## Dataset
The notebook loads this CSV file:

`RealEstate_HousePrice_Dataset_4200 - RealEstate_HousePrice_Dataset_4200.csv.csv`

The dataset has 4,200 house records. The target column is `house_price_inr`. The notebook uses `area_sqft` for Simple Linear Regression and all columns other than the target for Multiple Linear Regression and Polynomial Regression.

**Before running the notebook:** Place the CSV file in the same folder as the notebook, or update the path in the `pd.read_csv()` cell.

## Models Used
1. **Simple Linear Regression** — uses `area_sqft` as the single predictor.
2. **Multiple Linear Regression** — uses all columns except `house_price_inr` as predictors.
3. **Polynomial Regression (degree 3)** — expands the input features into polynomial features up to degree 3 and fits a linear regression model.
4. **Stochastic Gradient Descent** — uses scikit-learn's `SGDRegressor` to optimize a regression model.

## Evaluation Metrics
- **R² score:** Measures the proportion of target variance explained by the model. Higher is generally better.
- **MAE (Mean Absolute Error):** Average absolute prediction error.
- **MSE (Mean Squared Error):** Average squared prediction error.
- **RMSE (Root Mean Squared Error):** Square root of MSE, expressed in the target's units.

## Results Reported in the Notebook

| Model | Test R² score | Interpretation |
|---|---:|---|
| Simple Linear Regression | 0.5625 | Explains about 56.25% of the variation in house prices. |
| Multiple Linear Regression | 0.9177 | Explains about 91.77% of the variation in house prices. |
| Polynomial Regression (degree 3) | 0.9662 | Highest reported test R²; explains about 96.62% of the variation in house prices. |

For Simple Linear Regression, the notebook also reports:
- **MSE:** 66,989,260,021,849.47
- **MAE:** 6,294,593.70
- **RMSE:** 8,184,696.70

Polynomial Regression has the highest reported test R² score among the three models. However, a high test R² by itself does not prove that a model is free from overfitting; compare training and test performance before selecting a model for deployment.

## Gradient Descent Observation
The notebook's `SGDRegressor` predictions are extremely large negative values (around `-5 × 10^18`) and are not plausible house-price estimates. This suggests that the optimization did not converge to a useful solution with the current setup. The notebook does not report an SGD test R²/MAE/RMSE score.

To investigate, scale input features (for example, with `StandardScaler`), consider scaling the target, tune the learning rate and iteration settings, and evaluate the corrected model on held-out data. Ensure that preprocessing is fitted on training data only.

## Installation
Install the main Python dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

The notebook imports `root_mean_squared_error`, which requires a sufficiently recent version of scikit-learn. If that import fails, update scikit-learn:

```bash
pip install --upgrade scikit-learn
```

## How to Run
1. Keep the notebook and dataset CSV in the same folder (or update the dataset path).
2. Install the dependencies listed above.
3. Open the notebook in Jupyter Notebook, JupyterLab, or VS Code.
4. Run the cells from top to bottom.

## Important Notes
- The notebook includes an introductory explanation of supervised learning, regression versus classification, linear regression assumptions, the bias–variance trade-off, and overfitting/underfitting.
- The notebook's bias/variance section uses `1 - R²` and `R²` as simple proxies. These are not formal measurements of statistical bias and variance.
- The multiple-regression feature selection drops only the target column, so `house_id` is also included as a predictor. Consider removing identifier columns that do not represent meaningful predictive information.
- Compare train and test metrics, inspect residuals, and validate on new data before using predictions for real business decisions.

## Conclusion
The notebook reports that Multiple Linear Regression substantially improves on Simple Linear Regression, and degree-3 Polynomial Regression achieves the highest test R² score. The current `SGDRegressor` run produces unusable predictions and needs preprocessing and tuning before it can be compared fairly with the other models.
