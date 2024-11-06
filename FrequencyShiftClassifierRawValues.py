import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import zscore
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Read in the csv file that contains all trial data
dataFile = np.loadtxt('Data/magRaw.txt')
ampFile = np.loadtxt('Data/data1Amp.txt')

# Transpose so the labels form a column
dataFile = dataFile.T
ampFile = ampFile.T
                       
X = dataFile

formatted_matrix = np.where(X > 0, 1, np.where(X < 0, -1, 0))

print(formatted_matrix)


y = [1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8,9,9,9]

print(dataFile)
print(X)
print(y)

# Initialize the KNN Classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Perform cross-validated predictions
y_pred = cross_val_predict(knn, formatted_matrix, y, cv=3)

# Calculate and display the accuracy score
accuracy = accuracy_score(y, y_pred)
print(accuracy)
print(f"Cross-validated Accuracy: {accuracy:.2f}")

##knn.fit(formatted_matrix, y)
##predictions2 = knn.predict(formatted_matrix)
##accuracy2 = accuracy_score(y, predictions2)
##print(accuracy2)

# Calculate the confusion matrix
cmlabels = [1,2,3,4,5,6,7,8,9]
conf_matrix = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix (KNN Classification)")
plt.xticks(ticks=np.arange(len(cmlabels)) , labels=cmlabels )
plt.yticks(ticks=np.arange(len(cmlabels)) , labels=cmlabels)
plt.show()




                    

