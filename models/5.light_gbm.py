import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df = pd.get_dummies(df, columns=['Sex'])
    X = df.drop('Rings', axis=1).values
    y = df['Rings'].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return train_test_split(X, y, train_size=0.7, random_state=42)

def train_lightgbm(X_train, y_train, param_grid=None):
    model = lgb.LGBMRegressor(objective='regression', metric='rmse')
    if param_grid:
        grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        return best_model, grid_search.best_params_
    else:
        model.fit(X_train, y_train)
        return model, model.get_params()

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    n = len(y_test)  # number of samples
    k = X_test.shape[1]  # number of predictors

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
    model = model_class(**param_grid)
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5, scoring='neg_mean_squared_error', train_sizes=np.linspace(0.1, 1.0, 10)
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

# Main execution
file_path = 'abalone.csv'
X_train, X_test, y_train, y_test = load_and_prepare_data(file_path)

# LightGBM Tuned Model
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [-1, 10, 20],
    'learning_rate': [0.01, 0.1],
    'num_leaves': [31, 63]
}
best_model, best_params = train_lightgbm(X_train, y_train, param_grid)
print(f'Best Parameters: {best_params}')
y_pred = evaluate_model(best_model, X_test, y_test)

# Predictions for plotting
plot_results(y_test, y_pred, title='LightGBM: True Values vs Predictions')
plot_residuals(y_test, y_pred, title='LightGBM: Residuals Plot')
plot_histogram_of_residuals(y_test, y_pred, title='LightGBM: Histogram of Residuals')

# Plot learning curves for LightGBM model
print("\nPlotting Learning Curves for LightGBM Model:")
plot_learning_curves(X_train, y_train, X_test, y_test, lgb.LGBMRegressor, best_params)
