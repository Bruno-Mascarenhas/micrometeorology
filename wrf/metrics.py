import pandas as pd
import numpy as np 
from scipy.integrate import quad
from scipy.stats import ks_2samp, pearsonr
from sklearn.metrics import r2_score
#quad == mmethod to solve integrate equations

#mean bias difference
def MBD(obs,pred):
    return np.sum(pred - obs) / len(obs)

#root mean squared difference
def RMSD(obs,pred):
    return np.sqrt(np.sum(np.square(pred - obs))/len(obs))

#mean absolute difference
def MAD(obs,pred):
    return np.sum(np.abs(pred-obs))

#standard deviation of the residual
def SD(obs,pred):
    n = len(obs)
    f1 = np.sum(n*np.square(pred-obs))
    f2 = np.square(np.sum(pred-obs))
    return np.sqrt(f1-f2)/n

#coefficient of determination
def r2(obs,pred):
    return r2_score(obs,pred)

#r
def r(obs,pred):
    return pearsonr(obs,pred)[0]

#slope of best fit line
def SBF(obs,pred):
    pm = np.mean(pred)
    om = np.mean(obs)
    return np.sum((pred-pm)*(obs-om)) / np.sum(np.square(obs-om))

#uncertainty at 95%
def U95(obs,pred):
    return 1.96*np.sqrt((np.square(SD(obs,pred))+np.square(RMSD(obs,pred))))

#t-statistic
def TS(obs,pred):
    n = len(obs)
    return np.sqrt(((n-1)*np.square(MBD(obs,pred)))/(np.square(RMSD(obs,pred))-MBD(obs,pred)))

#Nash-Stutcliffe's efficiency
def NSE(obs,pred):
    om = np.mean(obs)
    return 1 - (np.sum(np.square(pred-obs))/np.sum(np.square(obs-om)))

#Willmott's index of agreement 
def WIA(obs,pred):
    om = np.mean(obs)
    f1 = np.sum(np.square(pred-obs))
    f2 = np.sum(np.square(np.abs(pred-om) + np.abs(obs-om)))
    return 1 - f1/f2 

#Legates's coefficient of efficiency
def LCE(obs,pred):
    om = np.mean(obs)
    return 1 - np.sum(np.abs(pred-obs))/np.sum(np.abs(obs-om))

#Kolmogorov-Smirnov statistic on 2 samples
def KSI(obs,pred):
    #wil return D and p-value, if p is high or D is small we can reject the hypothesis
    return ks_2samp(obs,pred)