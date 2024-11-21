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
from sklearn.preprocessing import PolynomialFeatures

# Data filenames
frshifts1 = 'Data/data1frshiftm.txt'
labels1 = 'Data/data1labelsm.txt'
frshiftsNsigns = 'Data/data1frshmagsig.txt' # Frequency shift magnitude, followed by sign binary values
frshsigns = 'Data/data1frshiftsigns.txt' # Binary sign values only
pressAmplitudes = 'Data/data1pressamp.txt' # Amplitude levels from pressed down state
frshifts2 = 'Data/data2frshiftm.txt' # 1, 2, 8, and 9 cm? 
frshsigns2 = 'Data/data2frshiftsigns.txt'
pressAmplitudes2 = 'Data/data2pressamp.txt'
labels2 = 'Data/data2labels.txt'
frshiftsDecimals = 'Data/data3decimalsfrsh.txt' # 1.5 to 8.5, 2 replication trials each 
labels3 = 'Data/data3labels.txt'
frshiftsFirstHalf = 'Data/data4frshfirsthalf.txt' # Integers 1 to 5, 4 replication trials each
labels4 = 'Data/data4labels.txt' # Labels for first half (1 to 5 cm)
frshiftsDecimalsFirstHalf = 'Data/data5decimalsfirsthalf.txt' # 1.5 to 5.5, 2 replication trials each
labels5 = 'Data/data5labels.txt' # Labels for first half decimals (1.5 to 5.5 cm)
fftIntsFirstHalf = 'Data/data7FFTintfirsthalf.txt'
fftDecsFirstHalf = 'Data/data7FFTdecimalfirsthalf.txt'
fftPowerIntHalf = 'Data/data8FFTintpowerfirsthalf.txt'
fftPowerDecHalf = 'Data/data8FFTdecpowerfirsthalf.txt'
frshInts2ndHalf = 'Data/data9frshInt2ndHalf.txt'
frshDec2ndHalf = 'Data/data9frshDec2ndHalf.txt'
fftPowerInt2ndHalf = 'Data/data10FFTintpower2ndhalf.txt'
fftPowerDec2ndHalf = 'Data/data10FFTdecpower2ndhalf.txt'

trial1 = [frshifts1, frshsigns, pressAmplitudes]
trial2 = [frshifts2, frshsigns2, pressAmplitudes2]

# Old control parameters
performingLeaveOut = False
leaveOut = 2
removeCols = []

# Filenames that are going to be used
dataFileName = fftPowerIntHalf
labelFileName = labels4
testDataFileName = fftPowerDecHalf
testLabelFileName = labels5

# Newer control parameters
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
    Xtest = np.hstack(X) 
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

# Take out rows when we're doing the leave out option
if (performingLeaveOut):
    for col_index, value in enumerate(y):  # matrix[-1] accesses the last row
        if value == leaveOut:
            removeCols.append(col_index)
    print("Column indices where value in bottom row is 1:", removeCols)
                       
    # Filter out the columns of the "leave 1 out" distance
    Xredu = np.delete(X, removeCols, 0);
    yredu = np.delete(y, removeCols);
    Xleftone = X[removeCols, :];

    print(Xleftone)

    #print(dataFile)
    print(Xredu)
    print(yredu)

# Regression models
# Simplify to just the frequency shift features now (no sign maps)
poly = PolynomialFeatures(5)

# Transform the features to polynomial features
X_poly = poly.fit_transform(X)
X_polytest = poly.fit_transform(Xtest)

# Train model
model = LinearRegression()
model.fit(X_poly, y)

# Make predictions
linregpred = model.predict(X_polytest)
print(linregpred)
mse = mean_squared_error(ytest, linregpred)
#r2 = r2_score(y, linregpred)
print(mse)
#print(r2)

# Regression models, with leave 1 out data 
# Transform the features to polynomial features
if (performingLeaveOut):
    Xpolyred = poly.fit_transform(Xredu)

    modelred = model.fit(Xpolyred, yredu)

    # Prepare training data (the left out data)
    Xleftonep = poly.fit_transform(Xleftone)
    leftoneLabels = [leaveOut, leaveOut, leaveOut];

    redpred = modelred.predict(Xleftonep)
    print(redpred)
    mse = mean_squared_error(leftoneLabels, redpred)
    r2 = r2_score(leftoneLabels, redpred)
    print(mse)
    print(r2)


# Scrapped code
##
### Read in the csv file that contains all trial data
##dataFile = np.loadtxt('Data/magRaw.txt')
##y = [1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8,9,9,9]
##
### Transpose so the labels form a column
##dataFile = dataFile.T
##
##X = dataFile
##
##
