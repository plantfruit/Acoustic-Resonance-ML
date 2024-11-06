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



leaveOut = 9
removeCols = []

# Read in the csv file that contains all trial data
dataFile = np.loadtxt('Data/magRaw.txt')
y = [1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8,9,9,9]

# Transpose so the labels form a column
dataFile = dataFile.T

X = dataFile

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
poly = PolynomialFeatures(2)

# Transform the features to polynomial features
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

linregpred = model.predict(X_poly)
print(linregpred)
mse = mean_squared_error(y, linregpred)
r2 = r2_score(y, linregpred)
print(mse)
print(r2)







