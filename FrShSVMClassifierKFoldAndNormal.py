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
trimic1_1 = 'Data/5x5_trimic1_1.txt' # Individual microphones' rows
trimic1_2 = 'Data/5x5_trimic1_2.txt'
trimic1_3 = 'Data/5x5_trimic1_3.txt'
trimic1_1and2 = 'Data/5x5_trimic1_1and2.txt' # Remove 1 microphone from the row
trimic1_2and3 = 'Data/5x5_trimic1_2and3.txt'
trimic1_1and3 = 'Data/5x5_trimic1_1and3.txt'
trimic1_1pulse = 'Data/5x5_trimic1_onepulse.txt' # Extract 1 pulse instead of 10 pulses
trimic1_1pulse_labels = 'Data/5x5_trimic1_onepulse_labels.txt'

miscobj2 = 'Data/miscobj2.txt'
miscobj2labels = 'Data/miscobj2_labels.txt'

<<<<<<< Updated upstream
# SELECT FILENAMES FOR ANALYSIS
fileName = miscobj2

labelFileName = miscobj2labels
=======
# Small array with 3 labels, and 3 "pulses per file," that is used to test the grouping function
groupingTest = 'Data/groupsorttest_features.txt' 
groupingTestLabels = 'Data/groupsorttest_labels.txt'

# 3x3 grid, pulse FFTs
g3x3_trimic1 = 'Data/3x3_trimic1.txt' # 15 files per label, groups of 5 trials that are "soft, "medium," and "hard" press
g3x3_trimic1_labels = 'Data/3x3_trimic1_labels.txt'

# SELECT FILENAMES FOR ANALYSIS
fileName = g3x3_trimic1

labelFileName = g3x3_trimic1_labels
>>>>>>> Stashed changes

# Read features and labels
X = np.loadtxt(fileName)
print(np.shape(X))
y = np.loadtxt(labelFileName)

# Reshape the 3D data to 2D (num_samples, num_channels * num_features)
X_reshaped = X


# CROSS-VALIDATED CLASSIFICATION
# Initialize the SVM classifier
svm_model = SVC(kernel='linear')  # You can change kernel here (e.g., 'rbf', 'poly')

# Perform cross-validation and get predictions for each sample
predictions = cross_val_predict(svm_model, X_reshaped, y, cv=5)

# Print predictions and true labels for each sample
#for i in range(len(predictions)):
#    print(f"Sample {i+1} - Predicted: {predictions[i]}, True: {y[i]}")

# Perform 5-fold cross-validation
accuracy = accuracy_score(y, predictions)
#cv_scores = cross_val_score(svm_model, X_reshaped, y, cv=5)


# Print the accuracy
print(accuracy)
#print(f"Accuracy for each fold: {cv_scores}")
#print(f"Average cross-validation accuracy: {cv_scores.mean() * 100:.2f}%")

# Generate the confusion matrix
cm = confusion_matrix(y, predictions)
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


# REGULAR CLASSIFICATION
# Train the SVM model
svm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test)

# Calculate the accuracy of the predictions
accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {accuracy * 100:.2f}%")

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(1, 26), yticklabels=np.arange(1, 26))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
