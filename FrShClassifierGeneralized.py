import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import zscore

frshifts1 = 'Data/data1frshiftm.txt'
labels1 = 'Data/data1labelsm.txt'
frshifts2 = 'Data/data1frshmagsig.txt' # Frequency shift magnitude, followed by sign binary values
frshsigns = 'Data/data1frshiftsigns.txt' # Binary sign values only
pressAmplitudes = 'Data/data1pressamp.txt' # Amplitude levels from pressed down state

# Control parameters
dataFileName = pressAmplitudes
labelFileName = labels1

numReplications = 4
combineVars = False
normalizeFeature = True
combineVarNames = [frshifts1, frshsigns, pressAmplitudes] 

# Read in the csv file that contains all trial data
# Assumes that the labels and the features are stored in separate files
dataFile = np.loadtxt(dataFileName)
labelFile = np.loadtxt(labelFileName)

X = []
if (combineVars):
    for varName in combineVarNames:
        featureTable = np.loadtxt(varName)
        if (featureTable.ndim < 2):
            featureTable = np.array([featureTable]).T
        # Normalize each data set separately
        if (normalizeFeature):
            featureTable = zscore(featureTable)
        X.append(featureTable)
    X = np.hstack(X)
else:    
    X = dataFile
    if (normalizeFeature):
        X = zscore(X)   

y = labelFile

print(X)
print(y)
    
# Initialize the KNN Classifier
knn = KNeighborsClassifier(n_neighbors=numReplications)

# Perform cross-validated predictions
y_pred = cross_val_predict(knn, X, y, cv=numReplications)

# Calculate and display the accuracy score
accuracy = accuracy_score(y, y_pred)

print(accuracy)
print(f"Cross-validated Accuracy: {accuracy:.2f}")

knn.fit(X, y)
predictions2 = knn.predict(X)
accuracy2 = accuracy_score(y, predictions2)
print("Non-cross validated KNN Accuracy")
print(accuracy2)

# Calculate the confusion matrix
cmlabels = [1,2,3,4,5,6,7,8,9]
conf_matrix = confusion_matrix(y, y_pred)
# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix (KNN Classification)")
plt.xticks(ticks=np.arange(len(cmlabels)) , labels=cmlabels )
plt.yticks(ticks=np.arange(len(cmlabels)) , labels=cmlabels)
plt.show()





