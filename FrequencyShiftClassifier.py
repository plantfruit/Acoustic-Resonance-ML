import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import zscore

# Read in the csv file that contains all trial data
dataFile = pd.read_csv('Data/data1n.csv')

# Transpose so the labels form a column
dataFile = dataFile.T
                       
X = dataFile.iloc[:, 1:].values # 16 predictor variables
y = dataFile.iloc[:, -1].values # Last column is the label

# Normalize with z-score
X = (X - X.mean()) / X.std()

print(dataFile)
print(X)
print(y)

# Initialize the KNN Classifier
knn = KNeighborsClassifier(n_neighbors=4)

# Perform cross-validated predictions
y_pred = cross_val_predict(knn, X, y, cv=4)

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix (KNN Classification)")
plt.show()

# Calculate and display the accuracy score
accuracy = accuracy_score(y, y_pred)
print(f"Cross-validated Accuracy: {accuracy:.2f}")
