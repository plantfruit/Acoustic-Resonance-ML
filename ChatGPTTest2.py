import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, ReLU, MaxPooling1D, Flatten, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Lambda
import numpy as np

from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, ReLU, MaxPooling1D, Flatten, Dense
import numpy as np

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

# Parameters
num_samples = 45        # Number of samples
num_channels = 3         # Number of FFT channels (microphones)
num_features = 251       # Number of FFT bins per channel
num_classes = 9          # Number of output classes

# Stack them along a new axis (axis=1) to create a 3D array
X = np.stack((file1, file2, file3), axis=1)  # shape: (num_samples, 3, 299)

# Example labels (binary classification in this case)
y = labelFile

# Define CNN model
inputs = Input(shape=(num_channels, num_features))  # Input shape: (3, 128)

y -= 1

# Custom layer to expand dimensions
class ExpandDimsLayer(Layer):
    def call(self, inputs):
        return tf.expand_dims(inputs, axis=-1)

# Initialize KFold with 5 splits (you can change this value)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store accuracy scores
accuracies = []

# Cross-validation loop
for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    # Separate convolutional layers for each channel
    conv_outputs = []
    for i in range(num_channels):
        # Use Lambda to extract the i-th channel
        channel_input = Lambda(lambda x: x[:, i, :])(inputs)  # Extract i-th channel (None, 128)
        
        # Apply the custom expand dimension layer
        channel_input = ExpandDimsLayer()(channel_input)  # Shape: (None, 128, 1)
        
        # Apply convolution, ReLU, max pooling, and flatten
        conv = Conv1D(filters=16, kernel_size=3, padding='same')(channel_input)
        conv = ReLU()(conv)
        conv = MaxPooling1D(pool_size=2)(conv)
        conv = Flatten()(conv)
        
        conv_outputs.append(conv)

    # Concatenate outputs from all channels
    concat = Concatenate()(conv_outputs)

    # Fully connected layers
    dense = Dense(64, activation='relu')(concat)
    dense = Dropout(0.5)(dense)  # Dropout for regularization
    outputs = Dense(num_classes, activation='softmax')(dense)

    # Create and compile the model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

     # Train the model
    model.fit(X_train, y_train, epochs=5, batch_size=16, verbose=0)

# Predict on the validation set
    predictions = model.predict(X_val)  # Get class probabilities
    predicted_classes = predictions.argmax(axis=-1) + 1  # Get the predicted class (add 1 to match original labels)

    # Print predictions and true labels for this fold
    print(predicted_classes)
   
    # Evaluate on the validation set
    val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    accuracies.append(val_accuracy[1])  # val_accuracy[1] is the accuracy metric
    

# Calculate average accuracy across all folds
average_accuracy = np.mean(accuracies)
print(f"Average cross-validation accuracy: {average_accuracy * 100:.2f}%")
