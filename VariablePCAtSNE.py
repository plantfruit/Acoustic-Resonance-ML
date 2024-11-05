import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import zscore
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Read in the csv file that contains all trial data
dataFile = np.loadtxt('Data/data1N.txt')
ampFile = np.loadtxt('Data/data1Amp.txt')

# Transpose so the labels form a column
dataFile = dataFile.T
ampFile = ampFile.T
                       
X = dataFile[:, :16] # 16 predictor variables
y = dataFile[:, -1] # Last column is the label

# Extract the 3 variables that we are trying to analyze
amplitudes = ampFile
magnitudes = dataFile[:, :8]
signs = dataFile[:, 8:16]

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

#print(tsneAmp.shape)
#print(tsneMag.shape)
#print(tsneSig.shape)

# PCA dimensionality reduction
pca = PCA()

pca.fit(amplitudes)
pcaAmp = pca.transform(amplitudes)
pca.fit(magnitudes)
pcaMag = pca.transform(magnitudes)
pca.fit(signs)
pcaSig = pca.transform(signs)

# t-SNE plotting code
# Credit: https://www.futurelearn.com/info/courses/machine-learning-for-image-data/0/steps/362489
plt.set_cmap('tab10')

fig, ax = plt.subplots(1,3, figsize = (18,6))
scatter1 = ax[0].scatter(tsneAmp[:,0], tsneAmp[:,1], c = y)
legend1 = ax[0].legend(*scatter1.legend_elements())
legend1.set_draggable(True)
ax[0].add_artist(legend1)
ax[0].set_title('t-SNE Amplitude')

scatter2 = ax[1].scatter(tsneMag[:,0], tsneMag[:,1], c = y)
legend2 = ax[1].legend(*scatter2.legend_elements())
legend2.set_draggable(True)
ax[1].add_artist(legend2)
ax[1].set_title('t-SNE Magnitude')

scatter3 = ax[2].scatter(tsneSig[:,0], tsneSig[:,1], c = y)
legend3 = ax[2].legend(*scatter3.legend_elements())
legend3.set_draggable(True)
ax[2].add_artist(legend3)
ax[2].set_title('t-SNE Sign')

#plt.figure()
#plt.scatter(tsneAmp[:,0], tsneAmp[:,1], c = y)

#plt.figure()
#plt.scatter(tsneMag[:,0], tsneMag[:,1], c = y)

#plt.figure()
#plt.scatter(tsneSig[:,0], tsneSig[:,1], c = y)

# PCA plotting code
fig2, ax2 = plt.subplots(1,3, figsize = (18,6))

scatter4 = ax2[0].scatter(pcaAmp[:,0], pcaAmp[:,1], c = y)
legend4 = ax2[0].legend(*scatter4.legend_elements())
legend4.set_draggable(True)
ax2[0].add_artist(legend4)
ax2[0].set_title('PCA Amplitude')

scatter5 = ax2[1].scatter(pcaMag[:,0], pcaMag[:,1], c = y)
legend5 = ax2[1].legend(*scatter5.legend_elements())
legend5.set_draggable(True)
ax2[1].add_artist(legend5)
ax2[1].set_title('PCA Magnitude')

scatter6 = ax2[2].scatter(pcaSig[:,0], pcaSig[:,1], c = y)
legend6 = ax2[2].legend(*scatter6.legend_elements())
legend6.set_draggable(True)
ax2[2].add_artist(legend6)
ax2[2].set_title('PCA Sign')

plt.show()

