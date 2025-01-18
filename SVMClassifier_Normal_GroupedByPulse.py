import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

mic1_1 = 'Data/trimic1_1.txt'
mic1_2 = 'Data/trimic1_2.txt'
mic1_3 = 'Data/trimic1_3.txt'
grid9_5samples = 'Data/bal2labels.txt'

trimic1 = 'Data/5by5_trimic1.txt' # 20 pulses per file
trimic1duplicate = 'Data/5by5_trimic1_possibleduplicate.txt'
trimic1labels = 'Data/5by5_trimic1_labels.txt'
trimic1re = 'Data/5x5_trimic1_re.txt' # Only 10 pulses per file
trimic1relabels = 'Data/5by5_trimic1_re_labels.txt'


# SELECT FILENAMES FOR ANALYSIS
fileName = trimic1re

labelFileName = trimic1relabels

# Read features and labels
X = np.loadtxt('Data/groupsorttest_features.txt')
print(np.shape(X))
y = np.loadtxt('Data/groupsorttest_labels.txt')

X_reshaped = X

# Create an array of group indices (0, 1, 2, ..., num_groups - 1)
# Example data: (num_samples, 3 channels, 128 features per channel)
num_groups = 9  # Number of groups
files_per_group = 2  # Files per group
total_samples = num_groups * files_per_group
group_indices = np.repeat(np.arange(num_groups), files_per_group)

# Split the data at the group level
from sklearn.model_selection import train_test_split

# Split group indices into train and test groups
train_groups, test_groups = train_test_split(
    np.unique(group_indices), test_size=0.2, random_state=35
)
print(train_groups.tolist())
print(test_groups.tolist())

# Filter data by group
# group_indices will always be larger than the 2nd parameter.
# So isin will "detect" which of the group indices have been placed in the train group
# and which have been placed in the test group.
train_mask = np.isin(group_indices, train_groups)
test_mask = np.isin(group_indices, test_groups)
print(train_mask.tolist())
print(test_mask.tolist())

# Using our knowledge of which group of indices is present in which class (train or test),
# we can form the train and test classes, but now they're organized by groups of
# pulses.
X_train, X_test = X_reshaped[train_mask], X_reshaped[test_mask]
print(X_train.tolist())
print(X_test.tolist())
y_train, y_test = y[train_mask], y[test_mask]
print(y_train.tolist())
print(y_test.tolist())

# Train the SVM model
svm_model = SVC(kernel='linear')  # You can change kernel here (e.g., 'rbf', 'poly')
svm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test)
print(y_pred.tolist())

# Calculate the accuracy of the predictions
accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {accuracy * 100:.2f}%")

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix using seaborn heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(1, 26), yticklabels=np.arange(1, 26))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
