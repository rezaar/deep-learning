import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow import keras
from keras.models import Sequential
from keras.layers import  Reshape, Conv2D, AveragePooling2D, Flatten, Dense
from keras.layers import Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l1_l2

#خواندن داده ها
data = pd.read_excel('mlp1.xlsx')
#جداسازی ویژگی و هدف
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

#حذف داده های گم شده
data = data.dropna()

#شناسایی و مدیریت مقادیر پرت 
scaler = RobustScaler(quantile_range=(5, 95))
x_scaled = scaler.fit_transform(x)
#تقسیم داده به آموزش ،اعتبار سنجی و آزمون
x_train, x_temp, y_train, y_temp = train_test_split(x_scaled, y, test_size=0.4, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)
    

model =Sequential()
Conv2D(filters=6, kernel_size=(5, 5), activation='relu', input_shape=(x_train.shape[1], 1, 1))
AveragePooling2D(pool_size=(2, 2))
Conv2D(16, (5, 5), activation='relu')
AveragePooling2D(pool_size=(2, 2))
Flatten()
Dense(120, activation='relu')
Dense(84, activation='relu')
Dense(1) # Output layer for regression

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

# Reshape data for CNN
X_train_reshaped = x_train.reshape(x_train.shape[0], x_train.shape[1], 1, 1)
X_val_reshaped = x_val.reshape(x_val.shape[0], x_val.shape[1], 1, 1)
X_test_reshaped = x_test.reshape(x_test.shape[0], x_test.shape[1], 1, 1)

# Train the model
#early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train_reshaped, y_train,
                    validation_data=(X_val_reshaped, y_val),
                    epochs=50,
                    batch_size=32,
                    #callbacks=[early_stopping],
                    verbose=1)

# 3. Model Evaluation and Analysis
y_pred = model.predict(X_test_reshaped).flatten()
# Visualize the loss curve during training
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()