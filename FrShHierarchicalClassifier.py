import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import zscore
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

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

trial1 = [frshifts1, frshsigns, pressAmplitudes]
trial2 = [frshifts2, frshsigns2, pressAmplitudes2]

# Filenames that are going to be used
dataFileName = fftPowerInt1stHalf
testDataFileName = fftPowerDec1stHalf

labelFileName = labelsInt1stHalf
testLabelFileName = labelsDec1stHalf

# Newer control parameters
numReplications = 4
polynomialRegressionDegree = 4
combineVars = False 
normalizeFeature = True
conductLinearRegression = False
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

# 1st stage of hierarchical classifier (KNN classifier)

# Perform predictions for the regular classifier (separate training and test datasets)
knn = KNeighborsClassifier(n_neighbors=numReplications)
knn.fit(X, y)

# Calculate the confusion matrix for regular classification test
print(Xtest)
otherSetPreds = knn.predict(Xtest)
#otherSetAccuracy = accuracy_score(otherSetLabels, otherSetPreds)
print("Test Dataset KNN Accuracy")
#print(otherSetAccuracy)
print(otherSetPreds)

# 2nd stage of hierarchical classifier (Polynomial/linear regression classifier)

# Read in a different feature set for the hierarchial classifier
X = 
if (normalizeFeature):
        Xtest = zscore(Xtest)
Xtest = 
if (normalizeFeature):
        Xtest = zscore(Xtest)


# Simplify to just the frequency shift features now (no sign maps)
poly = PolynomialFeatures(polynomialRegressionDegree)

# Transform the features to polynomial features
firstSetPredsDims = otherSetPreds.shape
for i in range(firstSetPredsDims[0]):
    integerPrediction = otherSetPreds[i]
    Xsegment = []
    ysegment = []

    print('NEW BLOCK')

    # Read select parts of the feature data set
    # Handle boundary conditions
    if (integerPrediction == 1):
        Xsegment = X[0:8,:]
        ysegment = y[0:8]
        ysegment = [0,0,0,0,1,1,1,1]
    elif (integerPrediction == 5):
        Xsegment = X[12:21,:]
        ysegment = y[12:21]
        ysegment = [-1,-1,-1,-1,0,0,0,0]
    # "In the middle" conditions
    else:
        Xsegment = X[(int(integerPrediction) - 2) * 4:(int(integerPrediction) + 1) * 4,:]
        ysegment = y[(int(integerPrediction) - 2) * 4:(int(integerPrediction) + 1) * 4]
        ysegment = [-1,-1,-1,-1,0,0,0,0,1,1,1,1]
    

    Xtestsegment = Xtest[i, :]
    ytestsegment = ytest[i]
    
    print(integerPrediction)
    #print(Xsegment)
    #print(ysegment)
    #print(Xtestsegment)
    print(ytestsegment)    

    if (Xtestsegment.ndim < 2):
        Xtestsegment = Xtestsegment.reshape(1,-1)
    X_poly = poly.fit_transform(Xsegment)
    X_polytest = poly.fit_transform(Xtestsegment)

    # Train model
    model = LinearRegression()
    if (conductLinearRegression):
        model.fit(Xsegment, ysegment)
    else:
        model.fit(X_poly, ysegment)

    # Make predictions    
    inputData = []
    if (conductLinearRegression):
        inputData = Xtestsegment
    else:
        inputData = X_polytest
        
    linregpred = model.predict(inputData)
    print('prediction')
    print(linregpred)
#mse = mean_squared_error(ytestsegment, linregpred)
#r2 = r2_score(y, linregpred)
#print(mse)
#print(r2)




