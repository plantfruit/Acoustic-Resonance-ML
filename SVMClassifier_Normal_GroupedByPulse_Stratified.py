import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# FILENAMES
# Old data, 3 x 3 grid. Each microphone was recorded separately
mic1_1 = 'Data/trimic1_1.txt'
mic1_2 = 'Data/trimic1_2.txt'
mic1_3 = 'Data/trimic1_3.txt'
grid9_5samples = 'Data/bal2labels.txt'

# 5 x 5 grid
# Each row is a pulse FFT
# Rows are grouped sequentially by the file they were extracted from
# e.g. 20 rows were from the same file 
trimic1 = 'Data/5by5_trimic1.txt' # 20 pulses per file
trimic1duplicate = 'Data/5by5_trimic1_possibleduplicate.txt'
trimic1labels = 'Data/5by5_trimic1_labels.txt'
trimic1re = 'Data/5x5_trimic1_re.txt' # Only 10 pulses per file
trimic1relabels = 'Data/5by5_trimic1_re_labels.txt'

# Small array with 3 labels, and 3 "pulses per file," that is used to test the grouping function
groupingTest = 'Data/groupsorttest_features.txt'
groupingTestLabels = 'Data/groupsorttest_labels.txt'

# SELECT FILENAMES FOR ANALYSIS
fileName = trimic1re

labelFileName = trimic1relabels

# Read features and labels
X = np.loadtxt(fileName)
print(np.shape(X))
y = np.loadtxt(labelFileName)

X_reshaped = X

# Dataset Parameters
num_labels = 25
files_per_label = 10
rows_per_file = 100
total_files = num_labels * files_per_label
total_rows = total_files * rows_per_file

# Train-test split: First 80 rows/train, last 20 rows/test per label
train_indices = []
test_indices = []

for label in range(1, num_labels + 1):
    # Get all rows for this label
    label_rows = np.where(y == label)[0]

    # Split the indices: first 80 for training, last 20 for testing
    train_indices.extend(label_rows[:80])
    test_indices.extend(label_rows[80:])
    
    # Split the indices: 
    # First 20 rows and last 60 rows for training
    #train_indices.extend(label_rows[:20])
    #train_indices.extend(label_rows[40:])
    # 2nd set of 20 rows for testing
    #test_indices.extend(label_rows[20:40])

# Convert to arrays for indexing
train_indices = np.array(train_indices)
test_indices = np.array(test_indices)

# Split the dataset
X_train, X_test = X_reshaped[train_indices], X_reshaped[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

# Train the SVM model
svm_model = SVC(kernel='linear')  # You can change kernel here (e.g., 'rbf', 'poly')
svm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test)

# Calculate the accuracy of the predictions
accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {accuracy * 100:.2f}%")

# Generate the confusion matrix with fixed size
all_labels = np.arange(1, num_labels + 1)  # All possible labels from 1 to 25
cm = confusion_matrix(y_test, y_pred, labels=all_labels)

# Visualize the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=all_labels, yticklabels=all_labels)
plt.title('Confusion Matrix (Fixed Size)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
