import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import zscore

# Read in the csv file that contains all trial data
#dataFile = np.loadtxt('Data/data1pktrNums.txt')
#dataFile = np.loadtxt('Data/data1pkNumAmpAvg.txt') # Only initial amplitudes
dataFile = np.loadtxt('Data/data1pkNumAmpAvg2.txt') # Both pressdown and initial states
#dataFile = np.loadtxt('Data/data1pkNumAmpAvgPress.txt') # Only pressdown amplitudes
dataFile = np.loadtxt('Data/data1pkAmp1and9.txt')
#dataFile = np.loadtxt('Data/data1pkAmp1and9_trial2
dataFile = np.loadtxt('Data/data1Amp1and9.txt')
dataFile = np.loadtxt('Data/data1Amp1and9int.txt')
dataFile = np.loadtxt('Data/data1Amp1and9_unmerged.txt')
ampFile = np.loadtxt('Data/data1Amp.txt')

# Assumes that the labels and the features are stored in separate files

# Transpose so the labels form a column
dataFile = dataFile.T
ampFile = ampFile.T
                       
X = dataFile
#y = [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8,9,9,9,9]
#y = [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5]
#y = [1,1,1,1,9,9,9,9]
y = [1,1,1,9,9,9]

##X[:,1] = X[:,1] / max(X[:,1])
##X[:,2] = X[:,2] / max(X[:,2])
##X[:,3] = X[:,3] / max(X[:,3])
##X[:,4] = X[:,4] / max(X[:,4])
##X[:,5] = X[:,5] / max(X[:,5])
##X[:,6] = X[:,6] / max(X[:,6])
##X[:,7] = X[:,7] / max(X[:,7])

#X[:,1] = zscore(X[:,1])

#print(X)
#X = X.reshape(-1, 1)
#print(dataFile)
print(X)
print(y)

#print(X.shape)
#print(y.shape)

# Initialize the KNN Classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Perform cross-validated predictions
y_pred = cross_val_predict(knn, X, y, cv=3)

# Calculate and display the accuracy score
accuracy = accuracy_score(y, y_pred)
print(accuracy)
print(f"Cross-validated Accuracy: {accuracy:.2f}")

knn.fit(X, y)
predictions2 = knn.predict(X)
accuracy2 = accuracy_score(y, predictions2)
print("Non-cross validated KNN Accuracy")
print(accuracy2)


plt.figure()
plt.plot(X)
plt.show()

# Calculate the confusion matrix
cmlabels = [1,2,3,4,5,6,7,8,9]
conf_matrix = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix (KNN Classification)")
plt.xticks(ticks=np.arange(len(cmlabels)) , labels=cmlabels )
plt.yticks(ticks=np.arange(len(cmlabels)) , labels=cmlabels)
plt.show()





