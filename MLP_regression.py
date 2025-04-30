import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = pd.read_excel('mlp1.xlsx')  
data = data.dropna()  

data = data[(np.abs(data - data.mean()) <= (3 * data.std()))].dropna()

x = data.iloc[:, :-1]
y = data.iloc[:, -1]

scaler = StandardScaler()
x = scaler.fit_transform(x)

x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=42)
x_test, x_val, y_test, y_val = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

models = {
    'sgd_constant': MLPRegressor(hidden_layer_sizes = (100), solver = 'sgd', learning_rate='constant', learning_rate_init=0.01, momentum=0, max_iter=1000, random_state=42, verbose=False),
    'sgd_adaptive_momentum': MLPRegressor(hidden_layer_sizes = (100), solver = 'sgd', learning_rate='adaptive', learning_rate_init=0.01, momentum=0.9, max_iter=1000, random_state=42, verbose=False),
    'adan_constant': MLPRegressor(hidden_layer_sizes = (100), solver = 'adam', learning_rate='constant', learning_rate_init=0.01, max_iter=1000, random_state=42, verbose=False),
    'adan_adaptive': MLPRegressor(hidden_layer_sizes = (100), solver = 'adam', learning_rate='adaptive', learning_rate_init=0.01, max_iter=1000, random_state=42, verbose=False),
    'lbfgs': MLPRegressor(hidden_layer_sizes=(100) ,solver='lbfgs', max_iter=1000, random_state=42, verbose=False)
}

results = {}
for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MSE': mse, 'MAE': mae, 'R2':r2, 'predictions': y_pred}

print("results:\n")
for name, metrics in results.items():
    print(f"{name}: MSE={metrics['MSE']:.4f}, MAE={metrics['MAE']:.4f}, r2={metrics['R2']:.4f}")


model_names = list(results.keys())



for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred_train = model.predict(x_train)
    y_pred_val = model.predict(x_val)
    


plt.figure(figsize=(15, 10))

for i, name in enumerate(model_names):
    plt.subplot(2, 3, i + 1)
    y_pred = results[name]['predictions']
    plt.scatter(y_test, y_pred, alpha=0.5)  
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.xlabel('Actual')
    plt.ylabel('predicted')
    plt.title(f'{name}: Actual vs predicted')

plt.subplot(2, 3, 6)
for name in model_names:
    error = y_test - results[name]['predictions']
    plt.hist(error, bins=30, alpha=0.5, label=name)
plt.xlabel('prediction')
plt.ylabel('Frequency')
plt.title('Prediction Error Histogram')
plt.legend

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")
colors = ['blue', 'green', 'orange', 'purple', 'red']
for i, name in enumerate (model_names):
    model = results[name]['model']
    if hasattr(model, 'loss_curve_'):
        plt.plot(model.loss_curve_, lable=name, color=colors[i], linewidth=2)
plt.title('Loss Curves of All Models', fontsize=16)
plt.xlabel('Iterations', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.7)
plt.show()