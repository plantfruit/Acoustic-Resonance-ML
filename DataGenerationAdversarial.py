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

#-----------------------------------------------------------------------------------------------------------------------------------------------------------
# Parameters and Introductory Variables
#-----------------------------------------------------------------------------------------------------------------------------------------------------------

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
labelsInt1stHalf = 'Data/data4labels.txt' # Labels for first half (1 to 5 cm)
frshiftsDecimalsFirstHalf = 'Data/data5decimalsfirsthalf.txt' # 1.5 to 5.5, 2 replication trials each
labelsDec1stHalf = 'Data/data5labels.txt' # Labels for first half decimals (1.5 to 5.5 cm)

fftIntsFirstHalf = 'Data/data7FFTintfirsthalf.txt' # FFT amplitudes for 1 - 5 cm
fftDecsFirstHalf = 'Data/data7FFTdecimalfirsthalf.txt' # FFt amplitudes for 1.5 - 5.5 cm

fftPowerInt1stHalf = 'Data/data8FFTintpowerfirsthalf.txt' # FFT dot products for 1 - 5 cm
fftPowerDec1stHalf = 'Data/data8FFTdecpowerfirsthalf.txt' # FFT dot products for 1.5 - 5.5 cm
frshInt2ndHalf = 'Data/data9frshInt2ndHalf.txt' # Frequency shifts for 5 - 9 cm
frshDec2ndHalf = 'Data/data9frshDec2ndHalf.txt' # Frequency shifts for 5.5 - 8.5 cm
fftPowerInt2ndHalf = 'Data/data10FFTintpower2ndhalf.txt' # FFT dot products for 5 - 9 cm
fftPowerDec2ndHalf = 'Data/data10FFTdecpower2ndhalf.txt' # FFT dot products for 5.5 - 8.5 cm
labelsInt2ndHalf = 'Data/data9labels.txt' # Labels for 5 to 9 cm        (1 cm intervals for both of these label variables) 
labelsDec2ndHalf = 'Data/data10labels.txt' # Labels for 5.5 to 8.5 cm

# Sine fit parameters (period and phase offset) on the frequency shift curves
sineFitInt1stHalf = 'Data/data11SineFitInt1stHalf.txt'
sineFitInt2ndHalf = 'Data/data11SineFitInt2ndHalf.txt'
sineFitDec1stHalf = 'Data/data11SineFitDec1stHalf.txt'
sineFitDec2ndHalf = 'Data/data11SineFitDec2ndHalf.txt'

# "Revised" trials by removing outliers 
sineFitInt1stHalfRe = 'Data/data12SineFitInt1stHalfRe.txt'
sineFitDec1stHalfRe = 'Data/data12SineFitDec1stHalfRe.txt'
sineFitInt2ndHalfRe = 'Data/data12SineFitInt2ndHalfRe.txt'
sineFitDec2ndHalfRe = 'Data/data12SineFitDec2ndHalfRe.txt'

# Spline interpolation FFT values
interpSplInt1stHalf = 'Data/data13InterSplInt1stHalf.txt'
interpSplDec1stHalf = 'Data/data13InterSplDec1stHalf.txt'

trial1 = [frshifts1, frshsigns, pressAmplitudes]
trial2 = [frshifts2, frshsigns2, pressAmplitudes2]

frshFFTPow2ndHalf = [frshInt2ndHalf, fftPowerInt2ndHalf]
frshFFTPow2ndHalfTest = [frshDec2ndHalf, fftPowerDec2ndHalf]

# Old control parameters
performingLeaveOut = False
leaveOut = 2
removeCols = []

# Filenames that are going to be used
dataFileName = frshiftsFirstHalf

labelFileName = labelsInt1stHalf

# Read the datasets from filenames
dataFile = np.loadtxt(dataFileName)
labelFile = np.loadtxt(labelFileName)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------
# Synthetic Data Generation
#-----------------------------------------------------------------------------------------------------------------------------------------------------------
# Import necessary packages
import pandas as pd
from sklearn import cluster

#from ydata_synthetic.utils.cache import cache_file
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
from ydata_synthetic.synthesizers.regular import RegularSynthesizer

# Read the datasets from filenames
dataFile = np.loadtxt(dataFileName)
labelFile = np.loadtxt(labelFileName)

#Data processing and analysis
num_cols = dataFile.shape[1] #list(data.columns[ data.columns != 'Class' ])
cat_cols = []

#----------------------------
#    GAN Training
#----------------------------

#Define the Conditional GAN and training parameters
noise_dim = 32
dim = 128
batch_size = 128
beta_1 = 0.5
beta_2 = 0.9

log_step = 100
epochs = 2 + 1
learning_rate = 5e-4
models_dir = '../cache'

#Test here the new inputs
gan_args = ModelParameters(batch_size=batch_size,
                           lr=learning_rate,
                           betas=(beta_1, beta_2),
                           noise_dim=noise_dim,
                           layers_dim=dim)

train_args = TrainParameters(epochs=epochs,
                             cache_prefix='',
                             sample_interval=log_step,
                             label_dim=-1,
                             labels=(0,1))

#create a bining
fraud_w_classes['Amount'] = pd.cut(fraud_w_classes['Amount'], 5).cat.codes

#Init the Conditional GAN providing the index of the label column as one of the arguments
synth = RegularSynthesizer(modelname='cgan', model_parameters=gan_args)

#Training the Conditional GAN
synth.fit(data= dataFile, label_cols= labelFile, train_arguments=train_args, num_cols=num_cols, cat_cols=cat_cols)

#Saving the synthesizer
synth.save('frshInt1stHalf_cgan_model.pkl')

#Loading the synthesizer
synthesizer = RegularSynthesizer.load('frshInt1stHalf_cgan_model.pkl')

#Sampling from the synthesizer
cond_array = pd.DataFrame(100*[1])#, columns=['Class'])
# Synthesizer samples are returned in the original format (inverse_transform of internal processing already took place)
sample = synthesizer.sample(cond_array)

print(sample)
