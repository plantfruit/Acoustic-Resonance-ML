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

trimic1 = 'Data/5by5_trimic1.txt'
trimic1labels = 'Data/5by5_trimic1_labels.txt'



# SELECT FILENAMES FOR ANALYSIS
fileName = trimic1

labelFileName = trimic1labels

labelFile = np.loadtxt(labelFileName)

# Stack them along a new axis (axis=1) to create a 3D array
X = np.loadtxt(trimic1)
print(np.shape(X))


# Example labels (binary classification in this case)
y = labelFile

# Reshape the 3D data to 2D (num_samples, num_channels * num_features)
X_reshaped = X



# Initialize the SVM classifier
svm_model = SVC(kernel='linear')  # You can change kernel here (e.g., 'rbf', 'poly')

# Perform cross-validation and get predictions for each sample
predictions = cross_val_predict(svm_model, X_reshaped, y, cv=5)

# Print predictions and true labels for each sample
#for i in range(len(predictions)):
#    print(f"Sample {i+1} - Predicted: {predictions[i]}, True: {y[i]}")

# Perform 5-fold cross-validation
cv_scores = cross_val_score(svm_model, X_reshaped, y, cv=5)

# Print the accuracy for each fold
print(f"Accuracy for each fold: {cv_scores}")

# Print the average accuracy across all folds
print(f"Average cross-validation accuracy: {cv_scores.mean() * 100:.2f}%")

# Generate the confusion matrix
cm = confusion_matrix(y, predictions)

# Visualize the confusion matrix using seaborn heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(1, 26), yticklabels=np.arange(1, 26))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

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
