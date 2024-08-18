import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Define the Neural Network model
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df = pd.get_dummies(df, columns=['Sex'])
    X = df.drop('Rings', axis=1).values
    y = df['Rings'].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return train_test_split(X, y, train_size=0.7, random_state=42)

def train_model(model, X_train, y_train, num_epochs=100, batch_size=64, learning_rate=0.001):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)

        epoch_loss /= len(X_train)
        train_losses.append(epoch_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    return train_losses

def evaluate_model(model, X_test, y_test):
    model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)
        y_pred = y_pred_tensor.numpy()

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

def plot_learning_curves(train_losses, title='Learning Curves'):
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.show()

# Main execution
file_path = 'abalone.csv'
X_train, X_test, y_train, y_test = load_and_prepare_data(file_path)

input_dim = X_train.shape[1]
hidden_dim = 64
output_dim = 1

model = NeuralNetwork(input_dim, hidden_dim, output_dim)
train_losses = train_model(model, X_train, y_train, num_epochs=100, batch_size=64, learning_rate=0.001)

y_pred = evaluate_model(model, X_test, y_test)

# Predictions for plotting
plot_results(y_test, y_pred, title='Neural Network: True Values vs Predictions')
plot_learning_curves(train_losses, title='Neural Network: Learning Curves')
