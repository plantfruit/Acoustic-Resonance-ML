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
trimic1_force = 'Data/trimic1_force.txt'
trimic1_force_labels = 'Data/trimic1_force_labels.txt'

miscobj2 = 'Data/miscobj2.txt'
miscobj2labels = 'Data/miscobj2_labels.txt'

# SELECT FILENAMES FOR ANALYSIS
trainFilename = trimic1re
testFilename = trimic1_force

trainLabelFilename = trimic1relabels
testLabelFilename = trimic1_force_labels

#file1 = np.loadtxt(file1Name)
#file2 = np.loadtxt(file2Name)
#file3 = np.loadtxt(file3Name)

# PARAMETERS
num_labels = 25

# READ FILES
trainDataFile = np.loadtxt(trainFilename)
testDataFile = np.loadtxt(testFilename)

trainLabelFile = np.loadtxt(trainLabelFilename)
testLabelFile = np.loadtxt(testLabelFilename)

X_train = trainDataFile
X_test = testDataFile
y_train = trainLabelFile
y_test = testLabelFile

# Split the data into train and test sets (80% train, 20% test)
# X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

# CLASSIFY
# Initialize the SVM classifier
svm_model = SVC(kernel='linear')  # You can change kernel here (e.g., 'rbf', 'poly')

# Train the SVM model
svm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test)

# Calculate the accuracy of the predictions
accuracy = accuracy_score(y_test, y_pred)
print(y_test)
print(y_pred)
print(f"Test accuracy: {accuracy * 100:.2f}%")

# Generate the confusion matrix
all_labels = np.arange(1, num_labels + 1)  # All possible labels from 1 to 25
cm = confusion_matrix(y_test, y_pred, labels=all_labels)

# Visualize the confusion matrix using seaborn heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(1, 26), yticklabels=np.arange(1, 26))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Outtakes
# Stack them along a new axis (axis=1) to create a 3D array
#X = np.stack((file1, file2, file3), axis=1)  # shape: (num_samples, 3, 299)
#print(np.shape(X))

# Example labels (binary classification in this case)
#y = labelFile

# Reshape the 3D data to 2D (num_samples, num_channels * num_features)
#X_reshaped = X.reshape(X.shape[0], -1)  # Shape: (100, 384)
