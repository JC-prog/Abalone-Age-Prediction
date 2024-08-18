import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor

def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df = pd.get_dummies(df, columns=['Sex'])
    X = df.drop('Rings', axis=1).values
    y = df['Rings'].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return train_test_split(X, y, train_size=0.7, random_state=42)

# Ridge Regression Base Model
def train_ridge_base_model(X_train, y_train):
    base_model = Ridge()
    base_model.fit(X_train, y_train)
    return base_model

# Ridge Regression Tuned Model
def train_ridge_regression(X_train, y_train, alpha_range=(0.01, 1.0, 0.1)):
    model = Ridge()
    param_grid = {'alpha': np.arange(*alpha_range)}
    grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=5)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    return best_model, grid_search.best_params_

# Model Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    n = len(y_test)
    k = X_test.shape[1]

    def adjusted_r2_score(r2, n, k):
        return 1 - (1 - r2) * (n - 1) / (n - k - 1)

    adj_r2 = adjusted_r2_score(r2, n, k)

    print(f'Mean Squared Error (MSE): {mse:.4f}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
    print(f'Mean Absolute Error (MAE): {mae:.4f}')
    print(f'R-squared (R2): {r2:.4f}')
    print(f'Adjusted R-squared (Adjusted R2): {adj_r2:.4f}')

    return y_pred

def plot_results(y_test, y_pred, title='True Values vs Predictions'):
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(title)
    plt.show()

def plot_residuals(y_test, y_pred, title='Residuals Plot'):
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.3)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(title)
    plt.show()

def plot_histogram_of_residuals(y_test, y_pred, title='Histogram of Residuals'):
    residuals = y_test - y_pred
    plt.hist(residuals, bins=30, edgecolor='k')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.show()

def plot_learning_curves(X_train, y_train, X_test, y_test, model_class, param_grid, title='Learning Curves'):
    train_sizes, train_scores, test_scores = learning_curve(
        model_class(), X_train, y_train, cv=5, scoring='neg_mean_squared_error', train_sizes=np.linspace(0.1, 1.0, 10)
    )
    train_scores_mean = -train_scores.mean(axis=1)
    test_scores_mean = -test_scores.mean(axis=1)

    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')
    plt.xlabel('Training Size')
    plt.ylabel('Mean Squared Error')
    plt.title(title)
    plt.legend(loc='best')
    plt.show()

def train_decision_tree_with_params(X_train, y_train, param_grid):
    model = DecisionTreeRegressor(random_state=42)
    grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=5)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    return best_model, grid_search.best_params_

# Main execution
file_path = 'abalone.csv'
X_train, X_test, y_train, y_test = load_and_prepare_data(file_path)

# Train and evaluate the base model
print("Base Model Evaluation:")
base_model = train_ridge_base_model(X_train, y_train)
y_pred_base = evaluate_model(base_model, X_test, y_test)

# Predictions for plotting (Base Model)
plot_results(y_test, y_pred_base, title='Base Model: True Values vs Predictions')
plot_residuals(y_test, y_pred_base, title='Base Model: Residuals Plot')
plot_histogram_of_residuals(y_test, y_pred_base, title='Base Model: Histogram of Residuals')

# Plot learning curves for the base model
print("\nPlotting Learning Curves for Base Model:")
plot_learning_curves(X_train, y_train, X_test, y_test, Ridge, {'alpha': [1.0]})  # Using alpha=1.0 as base

# Decision Tree Tuning and Evaluation
param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}
print("\nDecision Tree Tuned Model Evaluation:")
decision_tree_tuned_model, best_dt_params = train_decision_tree_with_params(X_train, y_train, param_grid)
print(f'Best Parameters for Decision Tree: {best_dt_params}')

# Evaluate Decision Tree Tuned Model
y_pred_dt_tuned = evaluate_model(decision_tree_tuned_model, X_test, y_test)
plot_results(y_test, y_pred_dt_tuned, title='Decision Tree Tuned Model: True Values vs Predictions')
