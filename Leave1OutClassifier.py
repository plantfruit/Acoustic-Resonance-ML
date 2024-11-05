import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import zscore

leaveOut = 9
removeCols = []

# Read in the csv file that contains all trial data
dataFile = np.loadtxt('Data/data1N.txt')

for col_index, value in enumerate(dataFile[-1]):  # matrix[-1] accesses the last row
    if value == leaveOut:
        removeCols.append(col_index)

print("Column indices where value in bottom row is 1:", removeCols)

# Transpose so the labels form a column
dataFile = dataFile.T
                       
X = dataFile[:, :16] # 16 predictor variables
y = dataFile[:, -1] # Last column is the label

#print(X[:, 1])

# Filter out the columns of the "leave 1 out" distance
Xredu = np.delete(X, removeCols, 0);
yredu = np.delete(y, removeCols);
Xleftone = X[removeCols, :];

print(Xleftone)
                      
# Normalize with z-score
#X = (X - X.mean()) / X.std()

#print(dataFile)
print(Xredu)
print(yredu)

# Initialize the KNN Classifier
knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(Xredu, yredu)

# Make predictions
predictions = knn.predict(Xleftone)

print(predictions)

