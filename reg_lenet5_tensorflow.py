import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error ,r2_score
from keras.models import Sequential
from keras.layers import Conv1D ,MaxPool1D, Flatten, Dense
from keras.layers import Dropout, BatchNormalization
from keras.optimizers import Nadam, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.regularizers import l1_l2

#خواندن داده ها
data = pd.read_excel('mlp1.xlsx')

data.fillna(method='ffill', inplace=True)
#حذف مقادیر پرت
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

#جداسازی ویژگی و هدف
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
#نرمال سازی
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x_scaled = x_scaled.reshape((x_scaled.shape[0],x_scaled.shape[1],1))
#تقسیم داده ها
x_train, x_temp, y_train, y_temp = train_test_split(x_scaled, y, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)
#تعریف مدل
model = Sequential([
    Conv1D(filters=6, kernel_size=5, strides=1, activation='tanh', padding='same', input_shape=(x_train.shape[1], 1), kernel_regularizer=l1_l2(l1=0.001, l2=0.01)),
    BatchNormalization(),
    MaxPool1D(pool_size=2, strides=2),
    Dropout(0.2),

    Conv1D(filters=16, kernel_size=3, strides=1, activation='tanh', padding='valid', kernel_regularizer=l1_l2(l1=0.001, l2=0.01)),
    BatchNormalization(),
    MaxPool1D(pool_size=2, strides=2),
    Dropout(0.2),

    Flatten(),

    Dense(units=120, activation='tanh', kernel_regularizer=l1_l2(l1=0.001, l2=0.01)),
    Dropout(0.2),
    BatchNormalization(),

    Dense(units=84, activation='tanh',  kernel_regularizer=l1_l2(l1=0.001, l2=0.01)),
    Dropout(0.2),
    BatchNormalization(),

    Dense(units=1) 
])
#کامپایل مدل
optimizer = Nadam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])

# آموزش مدل
early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model_LeNet5.h5', monitor='val_loss', save_best_only=True, mode='min')
reduc_lc = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
history = model.fit(x_train, y_train,
                    validation_data=(x_val, y_val),
                    epochs=120,
                    batch_size=16,
                    callbacks=[early_stopping,model_checkpoint,reduc_lc],
                    verbose=1)

y_pred = model.predict(x_test).flatten()
errors = y_pred - y_test

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(6, 4))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual VS Predicted")
plt.grid(True)
plt.show()

plt.figure(figsize=(6, 4))
plt.hist(errors, bins=30, edgecolor='black')
plt.title("Histogeram of Error")
plt.xlabel("Error")
plt.show()

plt.figure(figsize=(6, 4))
#pip install statsmodel
sns.residplot(x=y_test, y=y_pred, lowess=True, color="orange")
plt.xlabel("Actual")
plt.ylabel("Residual")
plt.title("Residual Error Plot")
plt.grid(True)
plt.show()

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)

print(f" MSE:{mse:.4f}, MAE:{mae:.4f}, R2_score:{r2:.4f}, RMSE:{rmse:.4f}")