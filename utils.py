# Some functions/class covered in this document:
# (class) brainData, (function) sliding_window, (function) Hilbert, (function) fir_band_pass 

#New-re-framed work according to fieldtrip workflow
from scipy.io import loadmat
from scipy.signal import spectrogram
import pandas as pd
import numpy as np
import os, glob

from nltools import Brain_Data
from nltools.mask import create_sphere,expand_mask
from nltools.data import Brain_Data, Adjacency,Design_Matrix
from nltools.stats import fdr, threshold, fisher_r_to_z, one_sample_permutation
from nltools.prefs import MNI_Template

import matplotlib.pyplot as plt
import nibabel as nib
#%matplotlib inline
from sklearn.decomposition import FastICA
from nltools.stats import downsample
import seaborn as sns
import mne
from mne.viz import plot_raw
from copy import deepcopy
from mne.io.pick import pick_types, pick_channels

from mne.filter import resample,construct_iir_filter
from mne.baseline import rescale
from mne import make_fixed_length_events
from mne import Epochs

from nltools.prefs import MNI_Template

from scipy.stats import gamma

import numpy.polynomial.polynomial as poly
from mne.time_frequency import psd_array_multitaper
from scipy.integrate import simps
import timeit
#from scipy.signal import welch
from scipy.stats import spearmanr
from scipy import signal


from sklearn.metrics import pairwise_distances
from nilearn.plotting import plot_glass_brain, plot_stat_map
import tables
from mne.time_frequency import tfr_array_morlet,psd_array_welch
import nltools as nlt
import random
import re

from itertools import repeat
from scipy.stats import gamma
import os
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree

from scipy.signal import firwin,hilbert
import scipy.signal 

class brainData:
    
    def __init__(self, fmri_pathname, fmri_mask_regions):
        """
        pathname: iEEG pathname including file name
        fmri_pathname: fmri pathname only without file names
        fmri_mask_regions should be a list of numbers, eg [20,21]
        """
        #self.pathname = pathname
        self.fmri_pathname = fmri_pathname
        #self.vmPFC_chs = vmPFC_chs
        #self.auditory_chs = auditory_chs
        self.raw_data = None
        self.data = None
        self.sfreq = None
        self.total_chns = None
        self.bd_chs = None
        #self.data = mne.io.read_raw_fif(pathname)
        #self.sfreq = self.data.info['sfreq']
        self.raw_data_reref = None
        self.fmri_mask_regions = fmri_mask_regions
        #self.mask = Brain_Data(os.path.join(fmri_pathname, "k50_3mm_gm_split_cleaned.nii.gz"),mask = self.fmri_pathname+'gm_mask_3mm.nii.gz')
        self.fmrilist = self.find_fmrifiles(fmri_pathname) 
        self.mask = Brain_Data(os.path.join(fmri_pathname, "k50_3mm_gm_split_cleaned.nii.gz"),mask = fmri_pathname+'gm_mask_3mm.nii.gz')
        #self.raw_data = None
        #self.masked_data = self.load_fmri_maskregions(fmri_mask_regions)
#     @property
#     def data(self):
#         return mne.io.read_raw_fif(self.pathname)
        
#     @property
#     def fmrilist(self):
#         return self.find_fmrifiles(self.fmri_pathname)

#     @property
#     def mask(self):
#         return Brain_Data(os.path.join(self.fmri_pathname, "k50_3mm_gm_split_cleaned.nii.gz"),mask = self.fmri_pathname+'gm_mask_3mm.nii.gz')
    
    def load_data_fromfif(self,pathname):
        self.data = mne.io.read_raw_fif(pathname)
        self.sfreq = self.data.info['sfreq']        
    
    def load_rawdata_fromedf(self,pathname):
        raw = mne.io.read_raw_edf(pathname, preload=True)
        mne.set_log_level("WARNING")
        trig_ix = np.where(np.array(raw.ch_names)=='DC1')[0][0]
        sfreq0 = raw.info['sfreq']
        data, times = raw["DC1",:]
        firstdiff = np.where(np.diff(data.T.ravel())>2)[0] 
        #print(firstdiff)
        #2717.80
        moviestart_diff = np.where(np.diff(firstdiff) > sfreq0 * 2717.80)[0]
        print(moviestart_diff)
        moviestart = firstdiff[moviestart_diff][0]
        movieend = firstdiff[moviestart_diff+1][0]
        print("start in secs:",moviestart/sfreq0)
        print("end in secs:",movieend/sfreq0)
    
        raw_2 = raw.crop(tmin = moviestart / sfreq0, tmax = movieend / sfreq0)
      
        self.raw_data = raw_2
        self.sfreq = sfreq0
        
    def load_data_from_raw(self):
        
        bad_channels = self.bd_chs
        good_channels = [a for a in self.total_chns if a not in bad_channels]
        raw_21 = self.raw_data.pick(picks = good_channels)
        self.total_chns = good_channels
        self.data = self.raw_data_reref
        self.raw_data = None
    
    def delete_bad_chs(self,chs_name,bd_chs):
        """
        step 1 in preprocessing: detect and delete bad channels
        """
        print("You chose",chs_name,"as the index of the final desired channel name")
        ch_idx_end = self.raw_data.info['ch_names'].index(chs_name)
        print("the index of that chosen channel is,", ch_idx_end)
        # #Pick valid channels
        picks_1 = self.raw_data.info['ch_names'][0:ch_idx_end]
        raw_21 = self.raw_data.pick(picks = picks_1)
        
        #bad_channels = ['LAD10', 'LACD5', 'LACD4', 'RAD6', 'RPHD9', 'LID9']
        
        #good_channels = [a for a in picks_1 if a not in bd_chs]
        
        #raw_21 = raw_21.pick(picks = good_channels)
        self.total_chns = picks_1
        self.raw_data = raw_21
        self.bd_chs = bd_chs
 
    def apply_filter(self,lfreq = 0.1, hfreq = None):
        """
        Optional step 2: high pass filter data
        """
        self.raw_data.filter(l_freq = lfreq, h_freq = hfreq, method = 'fir',fir_design = "firwin",filter_length = '200s')
    
    def apply_notch(self):
        """
        Step 3: filter out harmonics 60Hz
        """
        self.raw_data.notch_filter(np.arange(60, 241, 60), fir_design='firwin')
    
    def save_data(self,savename):
        self.data.save(savename,overwrite=True)
    
    def apply_rereference(self):

        ch_total_names = self.total_chns
        #annots = raw_3.annotations
        picks_bd = [pk_names for pk_names in ch_total_names if pk_names not in self.raw_data.info['bads']]

        raw_4 = self.raw_data.copy()
        raw_4 = raw_4.pick(picks = picks_bd)
        data_m1 = raw_4.get_data()

        #Step 5
        electrode_names = ch_total_names

        electrode_dupl = list()
        for j in range(len(electrode_names)):
            sup = ''.join(i for i in electrode_names[j] if i.isalpha())
            electrode_dupl.append(sup)

        electrode_dupl = list(dict.fromkeys(electrode_dupl))
        print("Electrode canonical names:")

        print(electrode_dupl)

        new_data_m1 = data_m1.copy()

        for names in electrode_dupl:
            list_a = [a for a in picks_bd if names in a]
            both = set(list_a).intersection(picks_bd)
            indices_lists_data = np.sort([picks_bd.index(x) for x in both])
            data_slice = data_m1[indices_lists_data,:]
            elec_pos_num = []

            for ele in list_a:
                match = re.match(r"([a-z]+)([0-9]+)", ele, re.I)
                if match:
                    items = match.groups()
                elec_pos_num.append(int(items[1]))

            part_matrix = np.zeros((len(list_a),data_m1.shape[1]))

            for i, num in enumerate(elec_pos_num):

                if (num-1) in elec_pos_num and (num+1) in elec_pos_num:
                    if np.corrcoef(data_slice[i],data_slice[i-1])[0,1] >= 0.7:
                        vec1 = data_slice[i,:] - data_slice[i-1,:]
                        if np.corrcoef(data_slice[i],data_slice[i+1])[0,1] >= 0.7:
                            vec1 += 1/2*data_slice[i-1,:] - 1/2*data_slice[i+1,:]
                    elif np.corrcoef(data_slice[i],data_slice[i+1])[0,1] >= 0.7:
                        vec1 = data_slice[i,:] - data_slice[i+1,:]
                    else:
                        vec1 = data_slice[i,:]

                elif (num-1) in elec_pos_num:
                    if np.corrcoef(data_slice[i],data_slice[i-1])[0,1] >= 0.7:
                        vec1 = data_slice[i,:] - data_slice[i-1,:]
                    else:
                        vec1 = data_slice[i,:]

                elif (num+1) in elec_pos_num:
                    if np.corrcoef(data_slice[i],data_slice[i+1])[0,1] >= 0.7:
                        vec1 = data_slice[i,:] - data_slice[i+1,:]        
                    else:
                        vec1 = data_slice[i,:]

                else:
                    vec1 = data_slice[i,:]

                part_matrix[i,:] = vec1

            new_data_m1[indices_lists_data,:] = part_matrix
        print("final data matrix size:",new_data_m1.shape)
        info = mne.create_info(ch_names = picks_bd, sfreq=self.raw_data.info['sfreq'], ch_types='eeg')
        raw_31 = mne.io.RawArray(new_data_m1, info)
        self.raw_data_reref = raw_31
            
         
        
        
    def get_rawdata(self,ch_names):
        #Fetch data from selected channels only
        return self.data.get_data(picks = ch_names)
    
#     def set_rawdata(self,raw_data):            
#         self.raw_data = raw_data
    
    @property
    def masked_data(self):
        return self.load_fmri_maskregions(self.fmri_mask_regions)

    def find_fmrifiles(self,fmri_pathname):        
        list_files = os.listdir(fmri_pathname) 
        all_files = [a for a in list_files if "sub-" in a]
        #print(all_files)
        return(all_files)
    
    def load_fmri_maskregions(self,fmri_mask_regions):
        MSK = self.mask[fmri_mask_regions[0]]
        for i in range(1,len(fmri_mask_regions)):
            MSK += self.mask[fmri_mask_regions[i]]
        #print(self.fmrilist)
        data_m = Brain_Data(self.fmri_pathname + self.fmrilist[0])
        MSK.plot()
        masked_data = data_m.apply_mask(MSK)
        fmri_series1 = data_m.extract_roi(mask = MSK)
        #print(fmri_series1.shape)
        region_values = np.zeros(shape = (len(self.fmrilist),len(fmri_series1)))
        region_values[0] = fmri_series1

        for i in range(1,len(self.fmrilist)):
            print("loading:", self.fmri_pathname+self.fmrilist[i])
            br_dat = Brain_Data(self.fmri_pathname+self.fmrilist[i])
            region_values[i] = br_dat.extract_roi(mask = MSK)
        
        return(region_values)
        
        
    def compute_isc(self, data_m1, fmri_matrix):
        """
        data_m1: Resampled intracranial hrf EEG data for one patient, n_chs * n_time
        fmri_matrix: fmri matrix, n_subject*n_voxel
        """
        isc_lists = []
        for i in range(fmri_matrix.shape[0]):
            isc_lists.append(np.corrcoef(data_m1,fmri_matrix[i, 0:len(data_m1)])[0,1])
        return(isc_lists,np.mean(isc_lists))

            
    def welch_psd(self, ch_names , n_fft = 0.5, n_overlap = 0.25, freq_bands = [1,16,36,120], iEEGPCA = False): 
        """
        welch's method
        """
        raw_matrix_data = self.data.get_data(picks = ch_names)
        if iEEGPCA:
            pca = PCA(n_components=0.95)
            raw_matrix_data_sub = pca.fit_transform(np.transpose(raw_matrix_data))
            raw_matrix_data = np.transpose(raw_matrix_data_sub)
            del raw_matrix_data_sub,pca

        psdwelch, freqwelch = psd_array_welch(raw_matrix_data, sfreq = self.sfreq, fmin = 0, fmax = 150, n_fft = int(n_fft * self.sfreq), n_overlap = int(n_overlap * self.sfreq), n_jobs=2, average = None);

        idx_band_low = np.logical_and(freqwelch >= freq_bands[0], freqwelch <= freq_bands[1]);
        idx_band_high = np.logical_and(freqwelch >= freq_bands[2], freqwelch <= freq_bands[3]);
   
        curr_psd = psdwelch[:,idx_band_low,:]        
        bpvals_low = np.mean(curr_psd,axis = 1)        
        curr_psd_high = psdwelch[:,idx_band_high,:]
        bpvals_high = np.mean(curr_psd_high,axis = 1)
        
        #bpvals_db_low = np.expand_dims(np.transpose(np.mean(bpvals_low,axis = 0)),axis = 1)
        #bpvals_db_high = np.expand_dims(np.transpose(np.mean(bpvals_high,axis = 0)),axis = 1)

        return(bpvals_low,bpvals_high)

    
    def stft_psd(self, ch_names , rms_interval = 1, freq_bands = [1,16,36,120], iEEGPCA = False):
        """
        short time fourier transform
        """
        raw_matrix_data = self.data.get_data(picks = ch_names)
        if iEEGPCA:
            pca = PCA(n_components=0.95)
            raw_matrix_data_sub = pca.fit_transform(np.transpose(raw_matrix_data))
            raw_matrix_data = np.transpose(raw_matrix_data_sub)
            del raw_matrix_data_sub,pca
            
        f, timer, Zxx = stft(x = raw_matrix_data, fs = self.sfreq, nperseg = int(1 * self.sfreq), noverlap = int(0.5 * self.sfreq))

        alpha_dat = Zxx[:,np.logical_and(f >= freq_bands[0] , f <= freq_bands[1]),:]
        alpha_dat_mean = np.expand_dims(np.mean(np.mean(abs(alpha_dat),axis = 0),axis = 0), axis =1)

        gamma_dat = Zxx[:,np.logical_and(f >= freq_bands[2] , f <= freq_bands[3]),:]
        gamma_dat_mean = np.expand_dims(np.mean(np.mean(abs(gamma_dat),axis = 0),axis = 0), axis =1)

        return(alpha_dat_mean,gamma_dat_mean)
    
    
    def multitaper_psd(self,ch_names,rms_interval=1,rms_step=0.1,freq_bands=[1,16,36,120], iEEGPCA = False):
        """
        return desired power spectrum bands
        Note it only returns 2 bands
        """
        raw_matrix_data = self.data.get_data(picks = ch_names)
        if iEEGPCA:
            pca = PCA(n_components=0.95)
            raw_matrix_data_sub = pca.fit_transform(np.transpose(raw_matrix_data))
            raw_matrix_data = np.transpose(raw_matrix_data_sub)
            del raw_matrix_data_sub,pca

        windowed_100_data = sliding_window(raw_matrix_data,size= int(rms_interval * self.sfreq),stepsize= int(rms_step * self.sfreq), padded=True, pad_val=0)
        
        print("We obtained a windowed data of size",windowed_100_data.shape)

        add_factor = 0
        bpvals_low = np.zeros(shape = (windowed_100_data.shape[0],windowed_100_data.shape[1]))
        bpvals_high = np.zeros(shape = (windowed_100_data.shape[0],windowed_100_data.shape[1]))

        for i in range(0,windowed_100_data.shape[1]):
            
            psdmulti,freqmulti = psd_array_multitaper(windowed_100_data[:,i,:],self.sfreq, bandwidth = 4,
                                                     normalization = "full",verbose = 0,n_jobs=4);
            freq_reso = freqmulti[1] - freqmulti[0];
            idx_band_low = np.logical_and(freqmulti >= freq_bands[0], freqmulti <= freq_bands[1]);
            idx_band_high = np.logical_and(freqmulti >= freq_bands[2], freqmulti <= freq_bands[3]);

            for j in range(0,windowed_100_data.shape[0]):
                bp = np.mean(psdmulti[j][idx_band_low])
                bp2 = np.mean(psdmulti[j][idx_band_high])
                bpvals_low[j][i] = bp
                bpvals_high[j][i] = bp2

        return(bpvals_low, bpvals_high)
    
    def get_convolved(self,data_low, data_high, rms_interval):
        """
        Convolve data (ideally those power values) with hrf func
        """
        
        raw_matrix_data_short = np.transpose(data_low)
        raw_matrix_data_high = np.transpose(data_high)

        dm = Design_Matrix(raw_matrix_data_short, sampling_freq= 1/rms_interval)
        dm2 = Design_Matrix(raw_matrix_data_high, sampling_freq= 1/rms_interval)
        
        dm_v = dm.convolve(conv_func='hrf')
        dm_v2 = dm2.convolve(conv_func='hrf')
        
        return(dm_v,dm_v2)
        
        
    
    
    def find_coupling_multitaper_isc(self, ch_names, fmri_matrix, rms_interval = 1, convolve = True, fmri_freq = 2, 
                                     freq_bands = [1,16,36,120], permute = False, short = False, iEEGPCA = False):
        """
        ch_names: list of channel names
        sfreq: int, sampling freq
        fmri_matrix: np array of fmri data
        convolve: bool, whether convolve with hrf or not
        permute: perform circular permutation and output distribution
        short: trigger the option if you want omly convolved & resampled (to 0.5Hz) iEEG data
        iEEGPCA: whether to PCA channels or use raw channels
        
        output:
        isc_low: 
        """
        print("The fmri matrix has a shape of:",fmri_matrix.shape)
        print("for channels:",ch_names)

        bpvals_db_low,bpvals_db_high = self.multitaper_psd(ch_names = ch_names , rms_interval = rms_interval, freq_bands = freq_bands, iEEGPCA = iEEGPCA)

        print("first bp matrix", bpvals_db_low.shape)

        raw_matrix_data_short = np.transpose(bpvals_db_low)
        raw_matrix_data_high = np.transpose(bpvals_db_high)

        dm = Design_Matrix(raw_matrix_data_short, sampling_freq= 1/rms_interval)
        dm2 = Design_Matrix(raw_matrix_data_high, sampling_freq= 1/rms_interval)
        
        dm_down = resample(np.transpose(dm),down = int(1/rms_interval * fmri_freq))
    
        #Print the correlation of each component:
#         for i in range(dm_down.shape[0]):
#             print("subject ",i)
#             print("==================")
#             for j in range(fmri_matrix.shape[0]):
#                 curr_dm = dm_down[i,:]
#                 print(np.corrcoef(curr_dm,fmri_matrix[j, 0:len(curr_dm)])[0,1])
        distrubutions_low = []
        distrubutions_high = []
        if convolve:
            dm_v = dm.convolve(conv_func='hrf')
            dm_v2 = dm2.convolve(conv_func='hrf')
            alu2 =  np.mean(dm_v,axis = 1)
            alu3 = np.mean(dm_v2,axis = 1)
            print('alu2',alu2.shape)
            alu21 = resample(alu2.iloc[:],down = int(1/rms_interval * fmri_freq))
            alu31 = resample(alu3.iloc[:],down = int(1/rms_interval * fmri_freq))
            print("resampled alu21 size:",alu21.shape)
            pearson_val_low = []
            spearman_val_low = []
            pearson_val_high = []
            spearman_val_high = []
            spearman_p_low = []
            spearman_p_high= []
            if short:
                return(alu21,alu31)
            lis_low, isc_low = self.compute_isc(alu21,fmri_matrix)
            lis_high, isc_high = self.compute_isc(alu31,fmri_matrix)

            if permute:
                prng = np.random.RandomState(6)
                for num in range(50000):
                    
                    shift1 = prng.choice(np.arange(len(alu21)), size=1,
                                 replace=True)[0]
                        
                    shift2 = prng.choice(np.arange(len(alu31)), size=1,
                                 replace=True)[0]
                    
                    #print(shift)
                    #print(alu31.shape)
                    shifted_data_low = np.concatenate((alu21[-shift1:],alu21[:-shift1]))
                    shifted_data_high = np.concatenate((alu31[-shift2:],alu31[:-shift2]))
                    
                    distrubutions_low.append(self.compute_isc(shifted_data_low,fmri_matrix))
                    distrubutions_high.append(self.compute_isc(shifted_data_high,fmri_matrix))
                
            #distributions = np.row_stack((distrubutions_low,distrubutions_high))
                    
        else:
            dm_d = np.mean(dm,axis = 1)
            dm_d2 = np.mean(dm2,axis = 1)

            print('dm_d',dm_d.shape)
            alud1 = resample(dm_d.iloc[:],down = int(1/rms_interval * fmri_freq) )
            alud2 = resample(dm_d2.iloc[:],down = int(1/rms_interval * fmri_freq) )

            pearson_val_low = []
            spearman_val_low = []
            pearson_val_high = []
            spearman_val_high = []

            for i in range(int(fmri_matrix.shape[0]*0.75)):
                bts_idx = np.random.choice(fmri_matrix.shape[0], int(0.8*fmri_matrix.shape[0]), replace = True)
                bt_fmri = np.array([fmri_matrix[row] for row in bts_idx])
                fmri_averaged = np.mean(bt_fmri,axis = 0)
                pearson_val_low.append(np.corrcoef(alud1,fmri_averaged[0:len(alud1)])[0,1])
                pearson_val_high.append(np.corrcoef(alud2,fmri_averaged[0:len(alud2)])[0,1])
                spearman_val_low.append(spearmanr(alud1,fmri_averaged[0:len(alud1)])[0])
                spearman_val_high.append(spearmanr(alud2,fmri_averaged[0:len(alud2)])[0])
        
        #Calculate circulated permutation p values
        permuted_mean_low = [distrubutions_low[i][1] for i in range(len(distrubutions_low))]
        permuted_mean_high = [distrubutions_high[i][1] for i in range(len(distrubutions_high))]
        #plt.hist(s2distr_audi_high_meandistr)
        permu_pval_low = np.sum(permuted_mean_low <= isc_low) / 50000
        permu_pval_high = np.sum(permuted_mean_high >= isc_high) / 50000
        
        if convolve:
            return(isc_low,
                  isc_high,
                  alu21,
                  alu31,
                  permu_pval_low,
                  permu_pval_high, 
                  lis_low,
                  lis_high)

        else:
                return(pearson_val_low,
                spearman_val_low,
                pearson_val_high,
                spearman_val_high,
                alud1,
                alud2)
    
    def plot_wholebraincorr(self, audi_dat_low, return_betas = False, plot = True):
        """
        plot whole brain voxel correlation with convolved and resampled iEEG data
        audi_dat_low: convolved iEEG data, not just low data
        """
        all_files = self.find_fmrifiles(self.fmri_pathname)

        all_sub = []
        for f in all_files:
            #print(os.path.join(self.fmri_pathname, f))
            sub_dat = Brain_Data(os.path.join(self.fmri_pathname, f))
            z_sub_dat = sub_dat.copy()
            z_sub_dat.data = (sub_dat.data - sub_dat.mean().data)/sub_dat.std().data
            z_sub_dat.data = z_sub_dat.data[int(0//20)::,:]
            #for i in range(len(fmri_time)):
            #    del_lists = [*del_lists,*np.arange(int(fmri_time[i][0]/2),int(fmri_time[i][0]/2)+1,1)]
            #print(del_lists)
            #res = [] 
            #[res.append(x) for x in del_lists if x not in res] 
            #print(len(res))
            #print(region_values2.shape)
            #z_sub_dat.data = np.delete(z_sub_dat.data,res,0)

            if audi_dat_low.shape[0] < z_sub_dat.data.shape[0]:
                z_sub_dat = z_sub_dat[:len(audi_dat_low)]
            elif audi_dat_low.shape[0] > z_sub_dat.data.shape[0]:
                audi_dat_low = audi_dat_low[:z_sub_dat.data.shape[0]]

            z_sub_dat.X = pd.DataFrame((audi_dat_low - np.mean(audi_dat_low))/np.std(audi_dat_low))

            stats = z_sub_dat.regress()
            #print("stats beta", stats['beta'])
            all_sub.append(stats['beta'])
            
        all_betas = Brain_Data(all_sub)
        if plot:
            stats_l2 = all_betas.ttest(threshold_dict={'fdr':.01})
            stats_l2['thr_t'].plot()
            plt.show()
            stats['beta'].plot()
            plt.show()
        if return_betas:
            return(all_betas)
    
    
    def find_autocorr_wholesignal(self, ch_names, len_of_auto = 100, downsampling = 128):
        """
        Find autocorrelation for the signal for every  box
        """
        #data_m1 = raw_data_file.copy()
        raw_matrix_data = self.data.get_data(picks = ch_names)

        pca = PCA(n_components=0.95)
        raw_matrix_data_sub = pca.fit_transform(np.transpose(raw_matrix_data))
        raw_matrix_data = np.transpose(raw_matrix_data_sub)
        print(raw_matrix_data.shape)

        bpvals = resample(raw_matrix_data, down = self.sfreq / downsampling)

        R_matrix_low = np.zeros(shape = (len(names),len_of_auto))
        dict_autocorr = {}

        for rows in range(bpvals.shape[0]):
            block_i = self.sliding_window(bpvals[rows],size = int(20 * (downsampling)),stepsize = int(10 * (downsampling)))
            block_i_autocorr = np.zeros(shape = (len(block_i),len_of_auto))
            for blk in range(len(block_i)):
                curr_block = block_i[blk,:]

                curr_block = (curr_block - np.mean(curr_block)) / np.std(curr_block)
                #curr_block = np.log(curr_block)

                block_i_autocorr[blk,0] = 1
                for shift in range(1,len_of_auto):
                    block_i_autocorr[blk,shift] = np.corrcoef(curr_block[:-shift],curr_block[shift:])[0][1]

            dict_autocorr[str(rows)] = block_i_autocorr
            R_matrix_low[rows,:] = np.median(block_i_autocorr,axis = 0)

        return(R_matrix_low,dict_autocorr)
    
    def dfa(self, x, scale_lim=[5,9], scale_dens=0.25, show=False):
        """
        Detrended Fluctuation Analysis - measures power law scaling coefficient
        of the given signal *x*.
        More details about the algorithm you can find e.g. here:
        Hardstone, R. et al. Detrended fluctuation analysis: A scale-free 
        view on neuronal oscillations, (2012).
        Args:
        -----
          *x* : numpy.array
            one dimensional data vector
          *scale_lim* = [5,9] : list of length 2 
            boundaries of the scale, where scale means windows among which RMS
            is calculated. Numbers from list are exponents of 2 to the power
            of X, eg. [5,9] is in fact [10**5, 10**9].
            You can think of it that if your signal is sampled with F_s = 128 Hz,
            then the lowest considered scale would be 10**5/128 = 32/128 = 0.25,
            so 250 ms.
          *scale_dens* = 0.25 : float
            density of scale divisions, eg. for 0.25 we get 2**[5, 5.25, 5.5, ... ] 
          *show* = False
            if True it shows matplotlib log-log plot.
        Returns:
        --------
          *scales* : numpy.array
            vector of scales (x axis)
          *fluct* : numpy.array
            fluctuation function values (y axis)
          *alpha* : float
            estimation of DFA exponent
        """
        # cumulative sum of data with substracted offset
        y = np.cumsum(x - np.mean(x))
        scales = (2**np.arange(scale_lim[0], scale_lim[1], scale_dens)).astype(np.int)
        fluct = np.zeros(len(scales))
        # computing RMS for each window
        for e, sc in enumerate(scales):
            fluct[e] = np.sqrt(np.mean(calc_rms(y, sc)**2))
        # fitting a line to rms data
        coeff = np.polyfit(np.log(scales), np.log(fluct), 1)
        if show:
            fluctfit = 2**np.polyval(coeff,np.log(scales))
            plt.loglog(scales, fluct, 'bo')
            plt.loglog(scales, fluctfit, 'r', label=r'$\alpha$ = %0.2f'%coeff[0])
            plt.title('DFA')
            plt.xlabel(r'$\log_{10}$(time window)')
            plt.ylabel(r'$\log_{10}$<F(t)>')
            plt.legend()
            plt.show()
        return scales, fluct, coeff[0]

    def multi_dfa(self, x, scale_lim=[5,10], scale_dens=0.25, show=False):
        """
        Detrended Fluctuation Analysis - measures power law scaling coefficient
        of the given signal *x*.
        More details about the algorithm you can find e.g. here:
        Hardstone, R. et al. Detrended fluctuation analysis: A scale-free 
        view on neuronal oscillations, (2012).
        Args:
        -----
          *x* : numpy.array
            one dimensional data vector
          *scale_lim* = [5,9] : list of length 2 
            boundaries of the scale, where scale means windows among which RMS
            is calculated. Numbers from list are exponents of 2 to the power
            of X, eg. [5,9] is in fact [10**5, 10**9].
            You can think of it that if your signal is sampled with F_s = 128 Hz,
            then the lowest considered scale would be 10**5/128 = 32/128 = 0.25,
            so 250 ms.
          *scale_dens* = 0.25 : float
            density of scale divisions, eg. for 0.25 we get 2**[5, 5.25, 5.5, ... ] 
          *show* = False
            if True it shows matplotlib log-log plot.
        Returns:
        --------
          *scales* : numpy.array
            vector of scales (x axis)
          *fluct* : numpy.array
            fluctuation function values (y axis)
          *alpha* : float
            estimation of DFA exponent
        """
        # cumulative sum of data with substracted offset  
        Y = x - x.mean(axis=1, keepdims=True)
        Y_sum = np.cumsum(Y,axis = 1)

        scales = (2**np.arange(scale_lim[0], scale_lim[1], scale_dens)).astype(np.int)
        fluct = np.zeros(len(scales))
        # computing RMS for each window

        for e, sc in enumerate(scales):
            arr2 = np.apply_along_axis(calc_rms,1,Y_sum,sc)
            #print(arr2.shape)
            #print(arr2[3,:])
            a3dd = LA.norm(arr2,axis=0)
            a3dd1 = np.square(a3dd)
            #print(a3dd[0])
            #print(a3dd1.shape
            fluct[e] = np.sqrt(np.mean(a3dd1))
        # fitting a line to rms data
        coeff = np.polyfit(np.log(scales), np.log(fluct), 1)
        if show:
            fluctfit = 2**np.polyval(coeff,np.log(scales))
            plt.loglog(scales, fluct, 'bo')
            plt.loglog(scales, fluctfit, 'r', label=r'$\alpha$ = %0.2f'%coeff[0])
            plt.title('DFA')
            plt.xlabel(r'$\log_{10}$(time window)')
            plt.ylabel(r'$\log_{10}$<F(t)>')
            plt.legend()
            plt.show()
        return scales, fluct, coeff[0]
    
    
    #https://raphaelvallat.com/entropy/build/html/_modules/entropy/entropy.html
    eps = 1e-50
    def _embed(self, x, order=3, delay=1):
        """Time-delay embedding.
        Parameters
        ----------
        x : 1d-array
            Time series, of shape (n_times)
        order : int
            Embedding dimension (order).
        delay : int
            Delay.
        Returns
        -------
        embedded : ndarray
            Embedded time-series, of shape (n_times - (order - 1) * delay, order)
        """
        N = len(x)
        if order * delay > N:
            raise ValueError("Error: order * delay should be lower than x.size")
        if delay < 1:
            raise ValueError("Delay has to be at least 1.")
        if order < 2:
            raise ValueError("Order has to be at least 2.")
        Y = np.zeros((order, N - (order - 1) * delay))
        for i in range(order):
            Y[i] = x[i * delay:i * delay + Y.shape[1]]
        return Y.T


    def _app_samp_entropy(self, x, order, metric='chebyshev', approximate=True):
        """Utility function for `app_entropy`` and `sample_entropy`.
        """
        _all_metrics = KDTree.valid_metrics
        if metric not in _all_metrics:
            raise ValueError('The given metric (%s) is not valid. The valid '
                             'metric names are: %s' % (metric, _all_metrics))
        phi = np.zeros(2)
        r = 0.2 * np.std(x, axis=-1, ddof=1)

        # compute phi(order, r)
        _emb_data1 = self._embed(x, order, 1)
        if approximate:
            emb_data1 = _emb_data1
        else:
            emb_data1 = _emb_data1[:-1]
        count1 = KDTree(emb_data1, metric=metric).query_radius(emb_data1, r,
                                                               count_only=True
                                                               ).astype(np.float64)
        # compute phi(order + 1, r)
        emb_data2 = self._embed(x, order + 1, 1)
        count2 = KDTree(emb_data2, metric=metric).query_radius(emb_data2, r,
                                                               count_only=True
                                                               ).astype(np.float64)
        if approximate:
            phi[0] = np.mean(np.log(count1 / emb_data1.shape[0]))
            phi[1] = np.mean(np.log(count2 / emb_data2.shape[0]))
        else:
            phi[0] = np.mean((count1 - 1) / (emb_data1.shape[0] - 1))
            phi[1] = np.mean((count2 - 1) / (emb_data2.shape[0] - 1))
        return phi


    def _numba_sampen(self, x, mm=2, r=0.2):
        """
        Fast evaluation of the sample entropy using Numba.
        """
        n = x.size
        n1 = n - 1
        mm += 1
        mm_dbld = 2 * mm

        # Define threshold
        r *= x.std()

        # initialize the lists
        run = [0] * n
        run1 = run[:]
        r1 = [0] * (n * mm_dbld)
        a = [0] * mm
        b = a[:]
        p = a[:]

        for i in range(n1):
            nj = n1 - i

            for jj in range(nj):
                j = jj + i + 1
                if abs(x[j] - x[i]) < r:
                    run[jj] = run1[jj] + 1
                    m1 = mm if mm < run[jj] else run[jj]
                    for m in range(m1):
                        a[m] += 1
                        if j < n1:
                            b[m] += 1
                else:
                    run[jj] = 0
            for j in range(mm_dbld):
                run1[j] = run[j]
                r1[i + n * j] = run[j]
            if nj > mm_dbld - 1:
                for j in range(mm_dbld, nj):
                    run1[j] = run[j]
        m = mm - 1
        while m > 0:
            b[m] = b[m - 1]
            m -= 1

        b[0] = n * n1 / 2
        a = np.array([float(aa) for aa in a])
        b = np.array([float(bb) for bb in b])
        p = np.true_divide(a, b)
        return -log(p[-1])


    def app_entropy(self, x, order=2, metric='chebyshev'):
        """Approximate Entropy.
        Parameters
        ----------
        x : list or np.array
            One-dimensional time series of shape (n_times).
        order : int
            Embedding dimension. Default is 2.
        metric : str
            Name of the distance metric function used with
            :py:class:`sklearn.neighbors.KDTree`. Default is
            `Chebyshev <https://en.wikipedia.org/wiki/Chebyshev_distance>`_.
        Returns
        -------
        ae : float
            Approximate Entropy.
        Notes
        -----
        Approximate entropy is a technique used to quantify the amount of
        regularity and the unpredictability of fluctuations over time-series data.
        Smaller values indicates that the data is more regular and predictable.
        The value of :math:`r` is set to :math:`0.2 * \\texttt{std}(x)`.
        Code adapted from the `mne-features <https://mne.tools/mne-features/>`_
        package by Jean-Baptiste Schiratti and Alexandre Gramfort.
        References
        ----------
        Richman, J. S. et al. (2000). Physiological time-series analysis
        using approximate entropy and sample entropy. American Journal of
        Physiology-Heart and Circulatory Physiology, 278(6), H2039-H2049.
        https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html
        Examples
        --------
        >>> from entropy import app_entropy
        >>> import numpy as np
        >>> np.random.seed(1234567)
        >>> x = np.random.rand(3000)
        >>> print(app_entropy(x, order=2))
        2.0754913760787277
        """
        phi = self._app_samp_entropy(x, order=order, metric=metric, approximate=True)
        return np.subtract(phi[0], phi[1])



    def sample_entropy(self, x, order=2, metric='chebyshev'):
        """Sample Entropy.
        Parameters
        ----------
        x : list or np.array
            One-dimensional time series of shape (n_times).
        order : int
            Embedding dimension. Default is 2.
        metric : str
            Name of the distance metric function used with
            :py:class:`sklearn.neighbors.KDTree`. Default is
            `Chebyshev <https://en.wikipedia.org/wiki/Chebyshev_distance>`_.
        Returns
        -------
        se : float
            Sample Entropy.
        Notes
        -----
        Sample entropy is a modification of approximate entropy, used for assessing
        the complexity of physiological time-series signals. It has two advantages
        over approximate entropy: data length independence and a relatively
        trouble-free implementation. Large values indicate high complexity whereas
        smaller values characterize more self-similar and regular signals.
        The sample entropy of a signal :math:`x` is defined as:
        .. math:: H(x, m, r) = -log\\frac{C(m + 1, r)}{C(m, r)}
        where :math:`m` is the embedding dimension (= order), :math:`r` is
        the radius of the neighbourhood (default = :math:`0.2 * \\text{std}(x)`),
        :math:`C(m + 1, r)` is the number of embedded vectors of length
        :math:`m + 1` having a
        `Chebyshev distance <https://en.wikipedia.org/wiki/Chebyshev_distance>`_
        inferior to :math:`r` and :math:`C(m, r)` is the number of embedded
        vectors of length :math:`m` having a Chebyshev distance inferior to
        :math:`r`.
        Note that if ``metric == 'chebyshev'`` and ``len(x) < 5000`` points,
        then the sample entropy is computed using a fast custom Numba script.
        For other distance metric or longer time-series, the sample entropy is
        computed using a code from the
        `mne-features <https://mne.tools/mne-features/>`_ package by Jean-Baptiste
        Schiratti and Alexandre Gramfort (requires sklearn).
        References
        ----------
        Richman, J. S. et al. (2000). Physiological time-series analysis
        using approximate entropy and sample entropy. American Journal of
        Physiology-Heart and Circulatory Physiology, 278(6), H2039-H2049.
        https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html
        Examples
        --------
        Sample entropy with order 2.
        >>> from entropy import sample_entropy
        >>> import numpy as np
        >>> np.random.seed(1234567)
        >>> x = np.random.rand(3000)
        >>> print(sample_entropy(x, order=2))
        2.192416747827227
        Sample entropy with order 3 using the Euclidean distance.
        >>> from entropy import sample_entropy
        >>> import numpy as np
        >>> np.random.seed(1234567)
        >>> x = np.random.rand(3000)
        >>> print(sample_entropy(x, order=3, metric='euclidean'))
        2.7246543561542453
        """
        x = np.asarray(x, dtype=np.float64)
        if metric == 'chebyshev' and x.size < 5000:
            return self._numba_sampen(x, mm=order, r=0.2)
        else:
            phi = self._app_samp_entropy(x, order=order, metric=metric,
                                    approximate=False)
            
            return -np.log(np.divide(phi[1], phi[0]))    
        
    def rec_to_time(self, vals,fps):
        times = np.array(vals)/60./fps
        times = [str(int(np.floor(t))).zfill(2)+':'+str(int((t-np.floor(t))*60)).zfill(2) for t in times]
        return times
    
    def normalize1(self, x):
        return ((x - np.mean(x)) / np.std(x))

    
    def process_HMMinfo(self, ch_names, hmm_components = 4):
        
        
        v_power_1,v_power_2 = self.welch_psd(ch_names, freq_bands = [36,64,80,120], iEEGPCA = False)
        v_power_3,v_power_4 = self.welch_psd(ch_names, freq_bands = [6,14,20,36], iEEGPCA = False)
        print("v_power_1:",v_power_1.shape)
        v_power = np.concatenate((v_power_1,v_power_2,v_power_3, v_power_4), axis = 1)
        
        
        print("v_power:",v_power.shape)
        v_power_window = v_power
        #v_power_window = self.sliding_window(v_power, size = 150, stepsize = 75)

        ab2 = np.apply_along_axis(self.normalize1, 1, v_power_window)
        print("ab2:",ab2.shape)
        #ab2_sub0 = ab2[:,-1,:]
        #return(ab2)
        model = GaussianHMM(n_components = hmm_components, n_iter=1000, covariance_type="diag")
        model.fit(ab2)
        zz = model.predict(ab2)
        return(ab2)
    
    
    
def sliding_window(data, size, stepsize=1, padded=False, axis=-1, copy=True, pad_val = 0):

    modul = (data.shape[-1] - size) % stepsize

    if padded:
        if  modul!= 0:
            appd = int(((data.shape[-1] - size) // stepsize + 1)*stepsize + size - data.shape[-1])
            print("append is:",appd)
            shape2 = data.shape[:-1] + (appd,)
            
            if pad_val == 0:
                data_append = np.zeros(shape2)
            else:
                data_append = np.empty(shape2)
                data_append[:] = np.nan

            data = np.concatenate((data,data_append),axis = -1)
    #print(data)
    if axis >= data.ndim:
        raise ValueError(
            "Axis value out of range"
        )

    if stepsize < 1:
        raise ValueError(
            "Stepsize may not be zero or negative"
        )

    if size > data.shape[axis]:
        raise ValueError(
            "Sliding window size may not exceed size of selected axis"
        )

    shape = list(data.shape)
    shape[axis] = np.floor(data.shape[axis] / stepsize - size / stepsize + 1).astype(int)
    shape.append(size)


    strides = list(data.strides)
    strides[axis] *= stepsize
    strides.append(data.strides[axis])

    #print(strides)
    strided = np.lib.stride_tricks.as_strided(
        data, shape=shape, strides=strides
    )


    if copy:
        return strided.copy()
    else:
        return strided
    
    
#From https://github.com/scipy/scipy/issues/6324
def Hilbert(arr):
    """
    zero padded hilbert to speed up calculation
    """
    arr_2 = None
    
    for i in range(arr.shape[0]):
        signal = arr[i,:]
        padding = np.zeros(int(2 ** np.ceil(np.log2(len(signal)))) - len(signal))
        tohilbert = np.hstack((signal, padding))
        result = hilbert(tohilbert)
        result = result[0:len(signal)]
        result = result.reshape((1,len(result)))
        #print(result)
        if arr_2 is None:
            arr_2 = result
        else:
            arr_2 = np.concatenate([arr_2,result],axis = 0)
    return(arr_2)
    
    
    
        #arrays02[i,:] = result
    #return(result)    
    #arrays02 = np.zeros(arrays.shape)
#     for i in range(arrays.shape[0]):
#         signal = arrays[i,:]
#         padding = np.zeros(int(2 ** np.ceil(np.log2(len(signal)))) - len(signal))
#         tohilbert = np.hstack((signal, padding))
#         result = hilbert(tohilbert)
#         result = result[0:len(signal)]
#         arrays02[i,:] = result
#     return arrays02


def fir_band_pass(signal,sfreq,lower_bound,upper_bound):
    """
    example usage: fir_band_pass(array, 512, 6, 14), where array shape = (nchs, ntimes)
    """
    #Determine the length of filter length according to mike x cohen
    filt_len = int(np.ceil(2 * (sfreq / lower_bound)))
    
    if filt_len % 2 == 0:
        filt_len += 1
        
    filter_coefs = firwin(filt_len, (lower_bound, upper_bound), pass_zero=False, nyq = sfreq / 2)
    filtered = np.array([np.convolve(xi, filter_coefs, mode='same') for xi in signal])
    
    return(filtered)


def generate_logistics(logics,freq_statement):
    """
    generate logic statements
    """
    logic_lists = []
    for i in logics:
        logic_lists.append(np.logical_and(freq_statement > i[0], freq_statement < i[1]))
    
    if len(logics) > 1:
        criteria0 = np.logical_or(logic_lists[0],logic_lists[1])
        
        for i in range(2, len(logic_lists)):
            criteria0 = np.logical_or(criteria0,logic_lists[i])
        return(criteria0)
    
    else:
        return(logic_lists[0])

def rec_to_time(vals,fps):
    times = np.array(vals)/60./fps
    times = [str(int(np.floor(t))).zfill(2)+':'+str(int((t-np.floor(t))*60)).zfill(2) for t in times]
    return times
        
def remove_preced0(str1):
    """
    remove proceeding 0 for some certain strings,
    for example, A01 -> A1;
                 A00 -> A0`
                 A010 -> A10
    
    """
    return re.sub('(?<![0-9])0*(?=[0-9])', '', str1)

def get_mni_locations(filename, electrode_names):
    """
    Load mni locations based on the filename given.
    Params:
    filename: the name of the csv where locations are stored
    vmPFC_names: the name of the electrode you want to load. eg 'LACD10'
    """
    #Load csv file MNI coordinates
    mat = pd.read_csv(filename)

    #Remove ''
    mat['electrodes'] = mat['electrodes'].str.replace('\'', '')

    #display(mat)

    #coordinates - electrode position,x,y&z
    #label_names - label name for the specific electrode

    coordinates = mat[['x','y','z']] 
    _label_names = mat['electrodes']
    label_names = [remove_preced0(x) for x in _label_names]
    #print(coordinates)
    #print(label_names2)
    #print(type(coordinates.iloc[1,0]))
    #print(type(label_names[1]))
    #vmPFC_names = ["LOFCD1","LOFCD2","LOFCD3","ROFCD1","ROFCD2","ROFCD3","RACD3"]
    if electrode_names == "all":
        return (coordinates,label_names)
    else:
        idx_01 = [label_names.index(a) for a in electrode_names]
        #print(idx_01)
        extract_coord = coordinates.iloc[idx_01,]
        centroid = extract_coord.mean()
        return (extract_coord)
    
def generate_chsnames(chs_prenames, chs_numbers):
    """
    generate channel names so that you don't have to type by yourself.
    e.g print(generate_chsnames(['LSTGD',"RAHCD"],[[9,10,11,12,13,14],list(range(5,9))]))
    [['LSTGD9', 'LSTGD10', 'LSTGD11', 'LSTGD12', 'LSTGD13', 'LSTGD14'], ['RAHCD5', 'RAHCD6', 'RAHCD7', 'RAHCD8']]
    """
    return_lists = []
    for i in range(len(chs_prenames)):
        chs_list = []
        for nums in chs_numbers[i]:
            chs_list.append(chs_prenames[i]+str(nums))
        return_lists.append(chs_list)
    return(return_lists)

def find_decayval(autocorr_array, threshold = 0.1):
    """
    Find the time for each autocorr_array row that drops to the certain threshold.
    """
    decays = []
    for i in range(autocorr_array.shape[0]):
        if autocorr_array[i,-1] > threshold:
            decays.append(len(autocorr_array[i,:]))
        else:
            decays.append(autocorr_array[i,:].tolist().index(min(autocorr_array[i,:], key = lambda x: abs(x-threshold))))
    return(decays)

def Hilbert_psd(subject, electrode_names, low_freq, high_freq, mode = "power"):
    """
    Calculate Hilbert transform power 
    example usage: test01 = Hilbert_psd(s2_data,s2auditory_names,6,14)
    """
    sfreq=subject.data.info['sfreq']
    picks_data = subject.data.get_data(picks = electrode_names)
    fir_filtered = fir_band_pass(picks_data, sfreq, low_freq, high_freq)
    hbert = Hilbert(fir_filtered)
    if mode == "phase":
        return(np.angle(hbert))
    elif mode == "power":
        return(abs(hbert))
    
# def Hilbert_psd(subject, electrode_names, low_freq, high_freq):
#     """
#     Calculate Hilbert transform power 
#     example usage: test01 = Hilbert_psd(s2_data,s2auditory_names,6,14)
#     """
#     sfreq=subject.data.info['sfreq']
#     picks_data = subject.data.get_data(picks = electrode_names)
#     fir_filtered = fir_band_pass(picks_data, sfreq, low_freq, high_freq)
#     hbert = Hilbert(fir_filtered)
#     return(abs(hbert))



def multitaper_psd(raw_data, ch_names, rms_interval=1, rms_step=0.5, freq_bands=None, iEEGPCA = False):
    """
    Newest multitaper version. Outputs nice results given a list of freq bands
    return desired power spectrum bands
    Example usage:
    test00a = multitaper_psd(s2_data.data, s2auditory_names,rms_interval=1, rms_step=0.5, freq_bands=freq_bands)
    
    """
    sfreq = raw_data.info['sfreq']
    raw_matrix_data = raw_data.get_data(picks = ch_names)
    
    
    if iEEGPCA:
        pca = PCA(n_components=0.95)
        raw_matrix_data_sub = pca.fit_transform(np.transpose(raw_matrix_data))
        raw_matrix_data = np.transpose(raw_matrix_data_sub)
        del raw_matrix_data_sub,pca

    windowed_100_data = sliding_window(raw_matrix_data,size=int(rms_interval*sfreq),stepsize=int(rms_step*sfreq), padded=True, pad_val=0)

    print("We obtained a windowed data of size",windowed_100_data.shape)

    psdmulti,freqmulti = psd_array_multitaper(windowed_100_data,sfreq,bandwidth=4,normalization="full",verbose = 0,n_jobs=4);
    freq_reso = freqmulti[1] - freqmulti[0];
    print("Your frequency resolution for the multitaper parameter is,",freq_reso, "hz")
    
    freq_list = []
    
    for freqs in freq_bands:
        
        idx_band_low = np.logical_and(freqmulti >= freqs[0], freqmulti <= freqs[1]);
        bp = np.mean(psdmulti[:,:,idx_band_low],axis=-1)
        freq_list.append(bp)

    return(freq_list)

def ITPC(phase_a,phase_b=None,mode="trial"):
    """
    Return the inter-trial phase clustering from phase input
    It has two modes,
    Input: phase_a: a phase of a signal
    Mode can be "trial" or "site"
    """
    if mode == "trial":
        ITPC = abs(np.mean(np.exp(1j*phase_a)))
    elif mode == "site":
        ITPC = abs(np.mean(np.exp(1j*(phase_a-phase_b))))
    return(ITPC)

# DISCLAIMER: This function is copied from https://github.com/nwhitehead/swmixer/blob/master/swmixer.py, 
#             which was released under LGPL. 
def resample_by_interpolation(signal, input_fs, output_fs):

    scale = output_fs / input_fs
    # calculate new length of sample
    n = round(len(signal) * scale)
    # use linear interpolation
    # endpoint keyword means than linspace doesn't go all the way to 1.0
    # If it did, there are some off-by-one errors
    # e.g. scale=2.0, [1,2,3] should go to [1,1.5,2,2.5,3,3]
    # but with endpoint=True, we get [1,1.4,1.8,2.2,2.6,3]
    # Both are OK, but since resampling will often involve
    # exact ratios (i.e. for 44100 to 22050 or vice versa)
    # using endpoint=False gets less noise in the resampled sound
    resampled_signal = np.interp(
        np.linspace(0.0, 1.0, n, endpoint=False),  # where to interpret
        np.linspace(0.0, 1.0, len(signal), endpoint=False),  # known positions
        signal,  # known data points
    )
    return resampled_signal

# Convert stuff into jpg
###############################################################
# import glob
# from pdf2image import convert_from_path

# gen_path = "F:\\ieeg_networks\\results\\centrality_vis\\"
# FilenamesList = glob.glob(gen_path+'*_glass.pdf')
# for file in FilenamesList:
#     pages = convert_from_path(file, 500)
#     for page in pages:
#         page.save(gen_path+file.split("\\")[-1].split(".")[0]+".jpg", 'JPEG')
############################################################
# Make gif from a series of jpgs
########################################################
# import glob
# from PIL import Image

# # filepaths
# fp_in = "F:/ieeg_networks/results/centrality_vis/*_glass.jpg"
# fp_out = "F:/ieeg_networks/results/centrality_vis/compiled.gif"

# # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
# img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
# img.save(fp=fp_out, format='GIF', append_images=imgs,
#          save_all=True, duration=200, loop=0)
##########################################################