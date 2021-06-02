"""
This script contains necessary information for morlet transformation of EEG signals
Author: Tiankang Xie
Liscence: MIT
"""

import os, glob
import pickle
import seaborn as sns
import numpy as np
import mne
from mne.viz import plot_raw,plot_raw_psd
from copy import deepcopy
from mne.io.pick import pick_types, pick_channels
from mne.filter import resample
from mne.baseline import rescale
from mne import make_fixed_length_events
from mne import Epochs
from mne.filter import filter_data

from scipy.stats import gamma

import numpy.polynomial.polynomial as poly
from mne.time_frequency import psd_array_multitaper,tfr_array_multitaper,psd_array_welch,tfr_array_morlet
from scipy.integrate import simps
from scipy.signal import hilbert
import timeit
from scipy.stats import spearmanr
from scipy import signal
from scipy.optimize import curve_fit

from sklearn.metrics import pairwise_distances
from mne.time_frequency import tfr_array_morlet 
import random

from numpy import linalg as LA

from utils import brainData, sliding_window, fir_band_pass, get_mni_locations, rec_to_time, Hilbert,find_decayval, Hilbert_psd

import matplotlib.transforms as transforms
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from joblib import Parallel, delayed, cpu_count
from sklearn.decomposition import PCA

import itertools

# New channels 
# s4_audi_idx=[196, 197, 198, 199, 101]
# s14_audi_idx = [143, 144, 26, 27, 40, 41, 42]

s2_audi_idx=[27, 97, 98, 99, 101]
s4_audi_idx=[196, 197, 198, 199, 201, 202, 101, 102]
#s5_audi_idx=[119, 120, 121, 122, 127, 128, 129, 130, 131]
s5_audi_idx=[119, 120, 121, 122, 127, 128, 129, 130]
s7_audi_idx=[47, 48, 49, 50, 51]
s10_audi_idx=[19, 81, 82, 83, 84]
#s14_audi_idx = [137, 138, 139, 143, 144, 24, 26, 27, 39, 40, 41, 42, 43, 52]
s14_audi_idx = [137, 138, 139, 143, 144, 24, 26, 27, 39, 40, 41, 42, 43]
s2_v_idx=[40, 41, 42, 43, 44, 120, 121, 122, 150, 151, 136]
s4_v_idx=[157, 158, 159, 171, 172, 55, 56, 57, 58, 59]
s5_v_idx=[29, 30, 31, 32, 33, 34, 134, 135, 136, 137, 150, 151]
s7_v_idx=[71, 72, 73, 74, 75, 104]
s10_v_idx=[30, 31, 32, 33, 99, 100, 40, 41, 42, 108, 109]
s14_v_idx = [107, 108, 109, 110, 123, 124]

eeg_filepath = "/dartfs/rc/lab/D/DBIC/cosanlab/Data/new_ieeg_analysis/data/iEEG/"

#=======================================================
#s2 scri[ts
bad_channels = []
ch_final = "F9"

fmri_path = "/dartfs/rc/lab/D/DBIC/cosanlab/Data/new_ieeg_analysis/data/fMRI/"

fmri_mask_regions_audi = [20,21]
fmri_mask_regions_v = [4,54,64,65]
fifpath = eeg_filepath+"raw_2_cleaned_v2.fif"

s2_data = brainData(fmri_pathname = fmri_path, fmri_mask_regions = fmri_mask_regions_audi)
s2_data.load_data_fromfif(fifpath)

s2_chs = s2_data.data.info['ch_names']
#=========================================================
# s4 Path
s4_amygdala = ["LAD1","LAD2","LAD3","LAD4","LAD5","RAD1","RAD2","RAD3","RAD4","RAD5"]

fmri_mask_regions_audi = [20,21]
fmri_mask_regions_v = [4,54,64,65]

fifpath = eeg_filepath+"raw_4_cleaned_v2.fif"

s4_data = brainData(fmri_pathname = fmri_path, fmri_mask_regions = fmri_mask_regions_audi)
s4_data.load_data_fromfif(fifpath)

s4_chs = s4_data.data.info['ch_names']

#=========================================================
#S5 Path
bad_channels = ['RID6', 'RPD5', 'RPD6', 'RPD7', 'RPD8', 'RPD9', 
                'RPHCD16', 'RPHCD15', 'RPHCD14', 'RPHCD2', 'RPHCD4', 'RPHCD3', 
                'RPHCD5', 'RPHCD1', 'RAHCD5', 'RAHCD4', 'RAHCD3', 'RAHCD2', 'RAHCD1']

ch_final = "C187"

fmri_mask_regions_audi = [20,21]
fmri_mask_regions_v = [4,54,64,65]

fifpath = eeg_filepath+"raw_5_cleaned_v2.fif"

s5_data = brainData(fmri_pathname = fmri_path, fmri_mask_regions = fmri_mask_regions_audi)
s5_data.load_data_fromfif(fifpath)

s5_chs = s5_data.data.info['ch_names']
#=========================================================
#S7 Path
bad_channels = ['LAD9', 'LACD8', 'LID2', 'LPHD11', 'LPHD12', 'LPHD10', 'LPHD13', 'LPHD14']
ch_final = "C121"

#path = "F:/Summer2019/s7_scripts/"
#fmri_path = "F:/Summer2019/fmri/"
fmri_mask_regions_audi = [20,21]
fmri_mask_regions_v = [4,54,64,65]
fifpath = eeg_filepath+"raw_7_cleaned_v2.fif"

s7_data = brainData(fmri_pathname = fmri_path, fmri_mask_regions = fmri_mask_regions_audi)
s7_data.load_data_fromfif(fifpath)

s7_chs = s7_data.data.info['ch_names']

#=========================================================
#S10 Path
fmri_mask_regions_audi = [20,21]
fmri_mask_regions_v = [4,54,64,65]

fifpath = eeg_filepath+"raw_10_cleaned_v2.fif"

s10_data = brainData(fmri_pathname = fmri_path, fmri_mask_regions = fmri_mask_regions_audi)
s10_data.load_data_fromfif(fifpath)

s10_chs = s10_data.data.info['ch_names']
#=========================================================
#S14 Path
fmri_mask_regions_audi = [20,21]
fmri_mask_regions_v = [4,54,64,65]

fifpath = eeg_filepath+"raw_14_cleaned_v3.fif"

s14_data = brainData(fmri_pathname = fmri_path, fmri_mask_regions = fmri_mask_regions_audi)
s14_data.load_data_fromfif(fifpath)

s14_chs = s14_data.data.info['ch_names']
#=======================================================================

#Feb 8 updated version. Very good

def _smooth_data(ieeg_data, mode = 'global_average'):
    """
    Simple algorithm to smooth out iEEG data
    """
    ieeg_data_copy = ieeg_data.copy()
    for i in range(ieeg_data.shape[0]):
        local_mean = np.mean(ieeg_data[i,:]) 
        local_std = np.std(ieeg_data[i,:])
        max_idx = np.where(abs(ieeg_data[i,:]) > local_mean+6.8*local_std)[0]
        #print(max_idx)
        msk_arr = np.ma.array(ieeg_data[i,:], mask=False)
        msk_arr.mask[max_idx] = True
        normal_global_mean = msk_arr.mean()

        for idx1 in max_idx:
            ieeg_data_copy[i,idx1] = normal_global_mean
            
    return ieeg_data_copy

s2_full_dat = s2_data.data.get_data()[:,int(0.24*512):-int(0.06*512)]
mid_len = int(s2_full_dat.shape[1]/2)
time_min = mid_len - 512*1358 + 70
time_max = mid_len + 512*1358 + 70
s2_dat = s2_full_dat[:,time_min:time_max]

#============================s4===============================================
s4_full_dat = s4_data.data.get_data()[:,int(0.24*512):-int(0.06*512)]
mid_len = int(s4_full_dat.shape[1]/2)
time_min = mid_len - 512*1358 + 37
time_max = mid_len + 512*1358 + 37
s4_dat = s4_full_dat[:,time_min:time_max]

#===============================s5==============================================
s5_full_dat = s5_data.data.get_data()
mid_len = int(s5_full_dat.shape[1]/2)
time_min = mid_len - 1024*1358 
time_max = mid_len + 1024*1358 
s5_dat = s5_full_dat[:,time_min:time_max]
s5_dat = mne.filter.resample(s5_dat,down=2) #Downsample

#===============================s7=================================================
s7_full_dat = s7_data.data.get_data()
mid_len = int(s7_full_dat.shape[1]/2)
time_min = mid_len - 512*1358 + 63
time_max = mid_len + 512*1358 + 63
s7_dat = s7_full_dat[:,time_min:time_max]

#==============================s10=====================================================
s10_full_dat = s10_data.data.get_data()
mid_len = int(s10_full_dat.shape[1]/2)
time_min = mid_len - 512*1358 + 42
time_max = mid_len + 512*1358 + 42
s10_dat = s10_full_dat[:,time_min:time_max]

#================================s14=====================================================
s14_full_dat = s14_data.data.get_data()[:,int(0.24*512):-int(0.06*512)]
mid_len = int(s14_full_dat.shape[1]/2)
time_min = mid_len - 512*1358 - 52#+ 15#- 83
time_max = mid_len + 512*1358 - 52#+ 15#- 83
s14_dat = s14_full_dat[:,time_min:time_max]
s14_dat = _smooth_data(s14_dat)
del s2_full_dat,s4_full_dat,s5_full_dat,s7_full_dat,s10_full_dat, s14_full_dat


freqs = np.logspace(*np.log10([1, 150]), num=30)
all_n_cycles = 7
n_cores = 16
write_path = "/dartfs/rc/lab/D/DBIC/cosanlab/Data/new_ieeg_analysis/results/morlet_results/"
test01a = tfr_array_morlet(np.expand_dims(s2_dat[s2_audi_idx,:],0), sfreq=512, freqs=freqs, n_cycles=all_n_cycles, output = 'complex',n_jobs=n_cores,decim=4)
test01v = tfr_array_morlet(np.expand_dims(s2_dat[s2_v_idx,:],0), sfreq=512, freqs=freqs, n_cycles=all_n_cycles, output = 'complex',n_jobs=n_cores,decim=4)
with open(write_path+'s2_audi_morlet_128.pickle', 'wb') as handle:
    pickle.dump(test01a, handle,protocol=4)
with open(write_path+'s2_vmpfc_morlet_128.pickle', 'wb') as handle:
    pickle.dump(test01v, handle,protocol=4)
del test01a,test01v

test02a = tfr_array_morlet(np.expand_dims(s4_dat[s4_audi_idx,:],0), sfreq=512, freqs=freqs, n_cycles=all_n_cycles, output = 'complex',n_jobs=n_cores,decim=4)
test02v = tfr_array_morlet(np.expand_dims(s4_dat[s4_v_idx,:],0), sfreq=512, freqs=freqs, n_cycles=all_n_cycles, output = 'complex',n_jobs=n_cores,decim=4)
with open(write_path+'s4_audi_morlet_128.pickle', 'wb') as handle:
    pickle.dump(test02a, handle,protocol=4)
with open(write_path+'s4_vmpfc_morlet_128.pickle', 'wb') as handle:
    pickle.dump(test02v, handle,protocol=4)
del test02a,test02v

test03a = tfr_array_morlet(np.expand_dims(s5_dat[s5_audi_idx,:],0), sfreq=512, freqs=freqs, n_cycles=all_n_cycles, output = 'complex',n_jobs=n_cores,decim=4)
test03v = tfr_array_morlet(np.expand_dims(s5_dat[s5_v_idx,:],0), sfreq=512, freqs=freqs, n_cycles=all_n_cycles, output = 'complex',n_jobs=n_cores,decim=4)
with open(write_path+'s5_audi_morlet_128.pickle', 'wb') as handle:
    pickle.dump(test03a, handle,protocol=4)
with open(write_path+'s5_vmpfc_morlet_128.pickle', 'wb') as handle:
    pickle.dump(test03v, handle,protocol=4)
del test03a,test03v

test04a = tfr_array_morlet(np.expand_dims(s7_dat[s7_audi_idx,:],0), sfreq=512, freqs=freqs, n_cycles=all_n_cycles, output = 'complex',n_jobs=n_cores,decim=4)
test04v = tfr_array_morlet(np.expand_dims(s7_dat[s7_v_idx,:],0), sfreq=512, freqs=freqs, n_cycles=all_n_cycles, output = 'complex',n_jobs=n_cores,decim=4)
with open(write_path+'s7_audi_morlet_128.pickle', 'wb') as handle:
    pickle.dump(test04a, handle,protocol=4)
with open(write_path+'s7_vmpfc_morlet_128.pickle', 'wb') as handle:
    pickle.dump(test04v, handle,protocol=4)
del test04a,test04v

test05a = tfr_array_morlet(np.expand_dims(s10_dat[s10_audi_idx,:],0), sfreq=512, freqs=freqs, n_cycles=all_n_cycles, output = 'complex',n_jobs=n_cores,decim=4)
test05v = tfr_array_morlet(np.expand_dims(s10_dat[s10_v_idx,:],0), sfreq=512, freqs=freqs, n_cycles=all_n_cycles, output = 'complex',n_jobs=n_cores,decim=4)
with open(write_path+'s10_audi_morlet_128.pickle', 'wb') as handle:
    pickle.dump(test05a, handle,protocol=4)
with open(write_path+'s10_vmpfc_morlet_128.pickle', 'wb') as handle:
    pickle.dump(test05v, handle,protocol=4)
del test05a,test05v

test06a = tfr_array_morlet(np.expand_dims(s14_dat[s14_audi_idx,:],0), sfreq=512, freqs=freqs, n_cycles=all_n_cycles, output = 'complex',n_jobs=n_cores,decim=4)
test06v = tfr_array_morlet(np.expand_dims(s14_dat[s14_v_idx,:],0), sfreq=512, freqs=freqs, n_cycles=all_n_cycles, output = 'complex',n_jobs=n_cores,decim=4)
with open(write_path+'s14_audi_morlet_128.pickle', 'wb') as handle:
    pickle.dump(test06a, handle,protocol=4)
with open(write_path+'s14_vmpfc_morlet_128.pickle', 'wb') as handle:
    pickle.dump(test06v, handle,protocol=4)
