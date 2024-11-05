import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import zscore

# Read in the csv file that contains all trial data
dataFile = np.loadtxt('Data/data1N.txt')

# Transpose so the labels form a column
dataFile = dataFile.T
                       
X = dataFile[:, :16] # 16 predictor variables
y = dataFile[:, -1] # Last column is the label

# Extract the 3 variables that we are trying to analyze
amplitudes = dataFile[:, :8]
magnitudes = abs(amplitudes)
signs = dataFile[:, 9:16]

print(amplitudes)
print(magnitudes)
print(signs)

#print(dataFile)

# Initialize the KNN Classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Perform cross-validated predictions
y_pred = cross_val_predict(knn, X, y, cv=4)

# Calculate the confusion matrix
cmlabels = [1,2,3,4,5,6,7,8,9]
conf_matrix = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix (KNN Classification)")
plt.xticks(ticks=np.arange(len(cmlabels)) , labels=cmlabels )
plt.yticks(ticks=np.arange(len(cmlabels)) , labels=cmlabels)
plt.show()

# Calculate and display the accuracy score
accuracy = accuracy_score(y, y_pred)
print(f"Cross-validated Accuracy: {accuracy:.2f}")
