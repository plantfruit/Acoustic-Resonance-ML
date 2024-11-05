import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import zscore
from sklearn.manifold import TSNE

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

allVars = np.concatenate((amplitudes, magnitudes, signs), 1)
print(allVars)

#print(dataFile)

# t-SNE dimensionality reduction
tsneAmp = TSNE().fit_transform(amplitudes)
tsneMag = TSNE().fit_transform(magnitudes)
tsneSig = TSNE().fit_transform(signs)

print(tsneAmp.shape)
print(tsneMag.shape)
print(tsneSig.shape)

# t-SNE plotting code
# Credit: https://www.futurelearn.com/info/courses/machine-learning-for-image-data/0/steps/362489
plt.set_cmap('tab10')

fig, ax = plt.subplots(1, 3, figsize = (20,5))
scatter1 = ax[0].scatter(tsneAmp[:,0], tsneAmp[:,1], c = y)
legend1 = ax[0].legend(*scatter1.legend_elements())
legend1.set_draggable(True)
ax[0].add_artist(legend1)

scatter2 = ax[1].scatter(tsneMag[:,0], tsneMag[:,1], c = y)
legend2 = ax[1].legend(*scatter2.legend_elements())
legend2.set_draggable(True)
ax[1].add_artist(legend2)

scatter3 = ax[2].scatter(tsneSig[:,0], tsneSig[:,1], c = y)
legend3 = ax[2].legend(*scatter3.legend_elements())
legend3.set_draggable(True)
ax[2].add_artist(legend3)

plt.show()
plt.close()
