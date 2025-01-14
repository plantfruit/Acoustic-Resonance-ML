import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Controllable constants
fftLength = 251 

mic1_1 = 'Data/trimic1_1.txt'
mic1_2 = 'Data/trimic1_2.txt'
mic1_3 = 'Data/trimic1_3.txt'

grid9_5samples = 'Data/bal2labels.txt'

# SELECT FILENAMES FOR ANALYSIS
file1Name = mic1_1
file2Name = mic1_2
file3Name = mic1_3

labelFileName = grid9_5samples

file1 = np.loadtxt(file1Name)
file2 = np.loadtxt(file2Name)
file3 = np.loadtxt(file3Name)

labelFile = np.loadtxt(labelFileName)

# Convert to numpy arrays
vec1 = file1
vec2 = file2
vec3 = file3

# Stack them along a new axis (axis=1) to create a 3D array
X = np.stack((vec1, vec2, vec3), axis=1)  # shape: (num_samples, 3, 299)

# Example labels (binary classification in this case)
y = labelFile

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the CNN model
model = Sequential()

# 1D Convolutional layer (treating the 3 vectors as "channels")
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(3, fftLength)))

# Global average pooling across the feature dimension (axis=2)
model.add(GlobalAveragePooling1D())

# Optionally, add a Dropout layer for regularization
model.add(Dropout(0.5))

# Fully connected layer for classification (output layer)
model.add(Dense(units=1, activation='sigmoid'))  # Binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=2, validation_data=(X_test, y_test))

# Evaluate the model
y_pred = (model.predict(X_test) > 0.5).astype("int32")  # Convert probabilities to binary labels
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
