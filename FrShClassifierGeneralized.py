import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import zscore

frshifts1 = 'Data/data1frshiftm.txt'
labels1 = 'Data/data1labelsm.txt'
frshiftsNsigns = 'Data/data1frshmagsig.txt' # Frequency shift magnitude, followed by sign binary values
frshsigns = 'Data/data1frshiftsigns.txt' # Binary sign values only
pressAmplitudes = 'Data/data1pressamp.txt' # Amplitude levels from pressed down state
frshifts2 = 'Data/data2frshiftm.txt'
frshsigns2 = 'Data/data2frshiftsigns.txt'
pressAmplitudes2 = 'Data/data2pressamp.txt'
labels2 = 'Data/data2labels.txt'

frshiftsDecimals = 'Data/data3decimalsfrsh.txt' # 1.5 to 8.5, 2 replication trials each 
labels3 = 'Data/data3labels.txt'

# Frequency shifts divided into the "two halves" of 10 cm tube
frshiftsFirstHalf = 'Data/data4frshfirsthalf.txt' # Integers 1 to 5, 4 replication trials each
labelsInt1stHalf = 'Data/data4labels.txt' # Labels for first half (1 to 5 cm)
frshiftsDecimalsFirstHalf = 'Data/data5decimalsfirsthalf.txt' # 1.5 to 5.5, 2 replication trials each
labelsDec1stHalf = 'Data/data5labels.txt' # Labels for first half decimals (1.5 to 5.5 cm)
frshInt2ndHalf = 'Data/data9frshInt2ndHalf.txt' # Frequency shifts for 5 - 9 cm
frshDec2ndHalf = 'Data/data9frshDec2ndHalf.txt' # Frequency shifts for 5.5 - 8.5 cm

# Area of FFT
fftPowerInt1stHalf = 'Data/data8FFTintpowerfirsthalf.txt' # FFT dot products for 1 - 5 cm
fftPowerDec1stHalf = 'Data/data8FFTdecpowerfirsthalf.txt' # FFT dot products for 1.5 - 5.5 cm
fftPowerInt2ndHalf = 'Data/data10FFTintpower2ndhalf.txt' # FFT dot products for 5 - 9 cm
fftPowerDec2ndHalf = 'Data/data10FFTdecpower2ndhalf.txt' # FFT dot products for 5.5 - 8.5 cm

# Merged the integer and decimal distances' frequency shifts for 1 - 5.5 cm
frshIntDec1stHalf = 'Data/data14frshIntDec1stHalf.txt'
labelsIntDec1stHalf = 'Data/data14labels.txt'

trial1 = [frshifts1, frshsigns, pressAmplitudes]
trial2 = [frshifts2, frshsigns2, pressAmplitudes2]

# Filenames that are going to be used
dataFileName = frshIntDec1stHalf
testDataFileName = frshIntDec1stHalf

labelFileName = labelsIntDec1stHalf
testLabelFileName = labelsIntDec1stHalf

numReplications = 4
combineVars = False
normalizeFeature = True
combineTrainData = trial1
combineTestData = trial2

# Read in the csv file that contains all trial data
# Assumes that the labels and the features are stored in separate files
dataFile = np.loadtxt(dataFileName)
labelFile = np.loadtxt(labelFileName)
testFile = np.loadtxt(testDataFileName)
otherSetLabels = np.loadtxt(testLabelFileName)

X = []
# Load and process the data
if (combineVars): # Combine multiple variable tables into 1 big table 
    for varName in combineTrainData:
        featureTable = np.loadtxt(varName)

        if (featureTable.ndim < 2):
            featureTable = np.array([featureTable]).T
        # Normalize each data set separately
        if (normalizeFeature):
            featureTable = zscore(featureTable)
        X.append(featureTable)
    X = np.hstack(X)
else: # Or just read all variables from 1 table
    if (dataFile.ndim < 2):
        dataFile = np.array([dataFile]).T
    X = dataFile
    if (normalizeFeature):
        X = zscore(X)
Xtest = []
if (combineVars): # Combine multiple variable tables into 1 big table 
    for varName in combineTestData:        
        featureTable = np.loadtxt(varName)

        if (featureTable.ndim < 2):
            featureTable = np.array([featureTable]).T
        # Normalize each data set separately
        if (normalizeFeature):
            featureTable = zscore(featureTable)
        Xtest.append(featureTable)
    Xtest = np.hstack(Xtest) 
else: # Or just read all variables from 1 table
    if (testFile.ndim < 2):
        testFile = np.array([testFile]).T
    Xtest = testFile
    if (normalizeFeature):
        Xtest = zscore(Xtest)

y = labelFile
ytest = otherSetLabels

print(X)
print(Xtest)
print(y)
print(otherSetLabels)
    
# Initialize the KNN Classifier
knnCV = KNeighborsClassifier(n_neighbors=numReplications)

# Perform cross-validated predictions
y_pred = cross_val_predict(knnCV, X, y, cv=numReplications)

# Calculate and display the accuracy score
accuracy = accuracy_score(y, y_pred)

print(accuracy)
print(f"Cross-validated Accuracy: {accuracy:.2f}")

# Perform predictions for the regular classifier (separate training and test datasets)
knn = KNeighborsClassifier(n_neighbors=numReplications)
knn.fit(X, y)

##predictions2 = knn.predict(X)
##accuracy2 = accuracy_score(y, predictions2)
##print("Non-cross validated KNN Accuracy")
##print(accuracy2)

# Calculate the confusion matrix for cross-validated test
cmlabels = [1,2,3,4,5,6,7,8,9]
conf_matrix = confusion_matrix(y, y_pred)
# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix (KNN Classification Cross Validation)")
plt.xticks(ticks=np.arange(len(cmlabels)) , labels=cmlabels )
plt.yticks(ticks=np.arange(len(cmlabels)) , labels=cmlabels)
plt.show()

# Calculate the confusion matrix for regular classification test
print(Xtest)
otherSetPreds = knn.predict(Xtest)
#otherSetAccuracy = accuracy_score(otherSetLabels, otherSetPreds)
print("Test Dataset KNN Accuracy")
#print(otherSetAccuracy)
print(otherSetPreds)
### Calculate the confusion matrix
##cmlabels_test = cmlabels
##conf_matrix_test = confusion_matrix(otherSetLabels, otherSetPreds)
### Plot the confusion matrix
##disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_test)
##disp.plot(cmap='Blues')
##plt.title("Confusion Matrix (KNN Classification of 2nd Data Set)")
##plt.xticks(ticks=np.arange(len(cmlabels_test)) , labels=cmlabels_test )
##plt.yticks(ticks=np.arange(len(cmlabels_test)) , labels=cmlabels_test)
##plt.show()






