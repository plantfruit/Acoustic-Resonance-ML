import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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

# Stack them along a new axis (axis=1) to create a 3D array
X = np.stack((file1, file2, file3), axis=1)  # shape: (num_samples, 3, 299)
print(np.shape(X))


# Example labels (binary classification in this case)
y = labelFile

# Reshape the 3D data to 2D (num_samples, num_channels * num_features)
X_reshaped = X.reshape(X.shape[0], -1)  # Shape: (100, 384)

# Split the data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

# Initialize the SVM classifier
svm_model = SVC(kernel='linear')  # You can change kernel here (e.g., 'rbf', 'poly')

# Train the SVM model
svm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test)

# Calculate the accuracy of the predictions
accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {accuracy * 100:.2f}%")

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix using seaborn heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(1, 10), yticklabels=np.arange(1, 10))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
