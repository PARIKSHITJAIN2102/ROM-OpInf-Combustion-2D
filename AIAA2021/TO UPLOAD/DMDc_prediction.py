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


A1 = np.load('A1dmdc.npy')
A2 = np.load('A2dmdc.npy')

A_final = np.float32(A1-A2)
U = np.load('U_DMDc.npy')

### get the projected data on U
data_projected = U.T@data

def predict_on_projected_data(data_projected, A_final, time_step):
    
    """
    Parameters
    ----------
    data = Scaled data projected on U
    
    A_final = loaded from the saved data of A1 and A2, A1-A2
    
    time_step = till which time step you need to predict 
    
    Returns
    -------
    Prediction of the projected data
    """
    
    #X_(t+1) = AX_(t)
    
    X0 = data_projected[:,0]
    X0 = np.reshape(X0,(np.shape(X0)[0],1))
    
    X_t = data_projected[:,0:time_step-1] #data till one time step before
    
    #A = np.linalg.multi_dot([U, A_final, U.T.conj()])
    
    tic = time.perf_counter()
    X_t_plus_1 = np.linalg.multi_dot(
                    [A_final, X_t]
                )
    
    X_pred = np.concatenate((X0 , X_t_plus_1), axis=1)
    toc = time.perf_counter()
    print(f"Prediction time is {toc - tic:0.4f} seconds")
    
    return X_pred

def get_original_data_from_projected_data(data_projected,U):
    
    """
    Parameters
    ----------
    data = Projected data 
    
    U = U from SVD saved in the directory 

    Returns
    -------
    Reconstructs original data from the projected data
    """
    O = np.linalg.multi_dot([U, data_projected])
    return O

# Calcuate the preidcitons 
Projected_data_predicted = predict_on_projected_data(data_projected, A_final, 60000)
Scaled_data_predicted = get_original_data_from_projected_data(Projected_data_predicted,U)
DMDc_final_pred_unscaled = unscale(Scaled_data_predicted, scales, variables=None)

##Get the data in the monitor locations 
MONITOR_LOCATIONS = config.MONITOR_LOCATIONS
O1 = pd.DataFrame(DMDc_final_pred_unscaled)

##Create traces of ROM at the monitor location
def getd(mon_loc, sli, df):
    """Returns
    -------
    Data only at the monitor locations
    """
    return np.array(df.loc[38523*sli + mon_loc,0::])


r = pd.DataFrame()
p = 0

for j in range(9):
    for i in MONITOR_LOCATIONS:
        r[p] = getd(i, j, O1)
        p = p+1

######################################## DMDc ROM results at the monitor locations 
traces_rom = r.T
########################################

#Save the DMDc result
np.save('traces_rom_DMDc_rsvd.npy', traces_rom)

