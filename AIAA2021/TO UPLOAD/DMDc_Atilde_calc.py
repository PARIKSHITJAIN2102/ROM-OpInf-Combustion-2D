import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.extmath import randomized_svd
import h5py
import matplotlib.pyplot as plt
import os


from config import *
#import utils
from data_processing import *
import time
time.perf_counter()

#Import time from the data or define it
t = np.arange(0.015,0.021,10**-7)

################################################################# Input data location
Base_folder = "C:\\Users\\Admin\\Desktop\\combustion\\" 
os.chdir(Base_folder)

Saved_scaled_data_file_name = Base_folder +'Scaled_data60k_final.npy'
Saved_scales = Base_folder + 'Scales_of_data60k_final.npy'
Saved_data = Base_folder + 'data60k_final.npy'
#Load the scaled data and scales
""" It is better to save the scaled data once and load it each time"""

data = np.load(Saved_scaled_data_file_name)
scales = np.load(Saved_scales)

""" OR """
"""Load the data and scale it"""
#data = np.load(Saved_data)
# data = lift(data)
# data, scales = scale(data, scales=None, variables=None)

##trainsize and r
r = 44
trainsize = 20000     # Number of snapshots used as training data.

###################################################### Extract traindata
traindata = data[0::,0:trainsize]



##get SVD
def get_SVD(data,r,t): 
    
    """
    Parameters
    ----------
    data = traindata
    r, t as defined in the input
    
    Returns
    -------
    Saves the singular value decomposition U, S, V from data = USV in the base_folder
    """
    rows = np.shape(data)[0]
    cols = np.shape(data)[1]
    
    dt = t[1] - t[0]
    X = data[:,0:cols-1]
    
    tic = time.perf_counter()
    u, s, VH_ = randomized_svd(X, n_components=500, n_iter = 10,random_state = 42)
    
    s = np.diag(s)
    
    
    U = u[:,0:r]
    S = s[0:r,0:r]
    V = VH_[:r,:].T

    
    toc = time.perf_counter()
    
    print(f"SVD calcuation time is {toc - tic:0.4f} seconds")
    
    np.save('U_DMDc.npy', U)
    np.save('S_DMDc.npy', S)
    np.save('V_DMDc.npy', V)
    
    return U,S,V

U,S,V = get_SVD(traindata,r,t)



def A1tide(data):
    
    """
    Parameters
    ----------
    data = X'-BU
    r, t as defined in the input
    
    Returns
    -------
    Atilde (Refer : https://arxiv.org/pdf/1409.6358.pdf)
    """
    
    U = np.load('U_DMDc.npy')
    S = np.load('S_DMDc.npy')
    V = np.load('V_DMDc.npy')
    cols = np.shape(data)[1]
    X1 = data[:,1:cols] #X' in discreet time
    
    tic = time.perf_counter()
    Atilde = U.T @ X1 @ V @ np.linalg.inv(S)
    toc = time.perf_counter()
    print(f"Atilde calcuation time is {toc - tic:0.4f} seconds")
    return Atilde

A1 = A1tide(traindata)

#Create control Matrix 

from config import *
BU = np.zeros(np.shape(traindata))
#for control boundary refer comp_domain.fig and grid.data
BU[np.arange(37512,38523,1),:] = np.float32(U(t[0:20000])) 
max_p = 1.5359*10**6
Scaled_BU = BU/max_p #as min P now is zero

#calcuating Atilde seperately to save memory
A2 = A1tide(Scaled_BU)
np.save('A1dmdc.npy', A1)
np.save('A2dmdc.npy', A2)
