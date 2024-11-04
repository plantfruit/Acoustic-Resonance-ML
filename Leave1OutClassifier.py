import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import zscore

# Read in the csv file that contains all trial data
dataFile = pd.read_csv('Data/data1without1.csv')

testData2cm = pd.read_csv('Data/data1only1.csv')

# Transpose so the labels form a column
dataFile = dataFile.T
testData2cm = testData2cm.T

# Extract feature and label columns
X = dataFile.iloc[:, 1:].values
y = dataFile.iloc[:, -1].values

# Initialize the KNN Classifier
knn = KNeighborsClassifier(n_neighbors=4)

# Train it with the set that doesn't have 2 cm
knn.fit(X, y)

# Predictions
predictions = knn.predict(testData2cm)
print(predictions)


# Calculate the confusion matrix
##labelOf2s = np.matrix([[2, 2, 2, 2]])
##conf_matrix = confusion_matrix(labelOf2s, predictions)
##
##disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
##disp.plot(cmap='Blues')
##plt.title("Confusion Matrix (KNN Classification)")
##plt.show()

# Calculate and display the accuracy score
accuracy = accuracy_score(y, y_pred)
print(f"Cross-validated Accuracy: {accuracy:.2f}")
