import os, json, h5py
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, iirdesign, zpk2tf, freqz
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import readligo as rl
import pandas as pd

plottype = 'png'

name= '1185619968'
fn_H1= 'DataFiles\H1_1185619968.hdf5'
fn_L1= 'DataFiles\L1_1185619968.hdf5'
fs= 4096
tevent= 1185619968
m1= 41.743
m2= 29.237
a1= 0.355
a2= -0.769
approx= 'lalsim.SEOBNRv2'
fband= [43.0, 300.0]
f_min= 10.0


strain1, time1, channel_dict1 = rl.loaddata(fn_H1, 'H1')
ts1 = time1[1] - time1[0] #-- Time between samples
fs1 = int(1.0 / ts1)          #-- Sampling frequency

strain2, time2, channel_dict2 = rl.loaddata(fn_L1, 'L1')
ts2 = time2[1] - time2[0] #-- Time between samples
fs2 = int(1.0 / ts2)          #-- Sampling frequency



segList1 = rl.dq_channel_to_seglist(channel_dict1['DEFAULT'], fs1)
length1 = 16  # seconds
strain_seg1 = strain1[segList1[0]][0:(length1*fs1)]
time_seg1 = time1[segList1[0]][0:(length1*fs1)]

segList2 = rl.dq_channel_to_seglist(channel_dict2['DEFAULT'], fs2)
length2 = 16  # seconds
strain_seg2 = strain2[segList2[0]][0:(length2*fs2)]
time_seg2 = time2[segList2[0]][0:(length2*fs2)]



strain_H1 = strain_seg1
time_H1 = time_seg1
chan_dict_H1 = channel_dict1

strain_L1 = strain_seg2
time_L1 = time_seg2
chan_dict_L1 = channel_dict2


# both H1 and L1 will have the same time vector, so:
time = time_H1
# the time sample interval (uniformly sampled!)
dt = time[1] - time[0]


deltat = 5.                     # seconds around the event
# index into the strain time series for this time interval:
indxt = np.where((time_H1 >= tevent-deltat) & (time_H1 < tevent+deltat))

def plot_strain():
    plt.figure()
    plt.plot(time_H1[indxt]-tevent, strain_H1[indxt], 'r', label='H1 strain', lw=1.2, alpha=0.6)
    plt.plot(time_L1[indxt]-tevent, strain_L1[indxt], 'g', label='L1 strain', lw=1.2, alpha=0.6)
    plt.grid(True)
    plt.xlim([3,3.1])
    #plt.ylim([-3,3])
    plt.xlabel('time (s) since '+str(tevent))
    plt.ylabel('strain')
    plt.legend(loc='best')
    plt.title(name +' Strain')
    plt.savefig("Strain Figs/" + name +'_Strain.'+plottype)

def plot_stretched_strain():
    plt.figure()
    plt.plot(time_H1[indxt]-tevent, strain_H1[indxt], 'r', label='H1 strain', lw=1.2, alpha=0.6)
    plt.plot(time_L1[indxt]-tevent, 4.5 * strain_L1[indxt], 'g', label='L1 strain', lw=1.2, alpha=0.6)
    plt.grid(True)
    plt.xlabel('time (s) since '+str(tevent))
    plt.ylabel('strain')
    plt.legend(loc='best')
    plt.title(name +' Stretched_Strain')
    plt.savefig("Strain Figs/" + name +'_Stretched_Strain.'+plottype)

def plot_H1_rolling_std():
    s1 = pd.Series(strain_H1[indxt])
    H1_rolling = s1.rolling(10).std()
    plt.figure()
    plt.plot(time_H1[indxt]-tevent, strain_H1[indxt], 'r', label='H1 strain', lw=1.2, alpha=0.6)
    plt.plot(time_H1[indxt]-tevent, H1_rolling, 'g', label='H1 rolling std', lw=1.2, alpha=0.6)
    #plt.plot(time_L1[indxt]-tevent, 4.5 * strain_L1[indxt], 'g', label='L1 strain', lw=1.2, alpha=0.6)
    
    plt.grid(True)
    plt.xlabel('time (s) since '+str(tevent))
    plt.ylabel('strain')
    plt.legend(loc='best')
    plt.title(name +' H1 STD Strain')
    plt.savefig("Strain Figs/" + name +'_H1_STD_Strain.'+plottype)

def plot_L1_rolling_std():
    s2 = pd.Series(strain_L1[indxt])
    L1_rolling = s2.rolling(10).std()
    plt.figure()
    plt.plot(time_L1[indxt]-tevent, strain_L1[indxt], 'r', label='L1 strain', lw=1.2, alpha=0.6)
    plt.plot(time_L1[indxt]-tevent, L1_rolling, 'g', label='L1 rolling std', lw=1.2, alpha=0.6) 
    plt.grid(True)
    plt.xlabel('time (s) since '+str(tevent))
    plt.ylabel('strain')
    plt.legend(loc='best')
    plt.title(name +' L1 STD Strain')
    plt.savefig("Strain Figs/" + name +'_STD_Stretched_Strain.'+plottype)
# number of sample for the fast fourier transform:
NFFT = 4*fs
f_min = 10
f_max = 2000
Pxx_H1, freqs = mlab.psd(strain_H1, Fs = fs, NFFT = NFFT)
Pxx_L1, freqs = mlab.psd(strain_L1, Fs = fs, NFFT = NFFT)

# We will use interpolations of the ASDs computed above for whitening:
psd_H1 = interp1d(freqs, Pxx_H1)
psd_L1 = interp1d(freqs, Pxx_L1)


def plot_ASD():
    plt.figure()
    plt.loglog(freqs, np.sqrt(Pxx_H1), 'r', label='H1 strain', lw=1.2, alpha=0.6)
    plt.loglog(freqs, np.sqrt(Pxx_L1), 'g', label='L1 strain', lw=1.2, alpha=0.6)
    plt.axis([f_min, f_max, 1e-24, 1e-19])
    plt.grid(True)
    plt.ylabel('ASD (strain/rtHz)')
    plt.xlabel('Freq (Hz)')
    plt.legend(loc='best')
    plt.title(name +' ASDs')
    plt.savefig("Strain Figs/" + name +'_ASDs.'+plottype)



# function to whiten data
def whiten(strain, interp_psd, dt):
    Nt = len(strain)
    freqs = np.fft.rfftfreq(Nt, dt)

    # whitening: transform to freq domain, divide by asd, then transform back, 
    # taking care to get normalization right.
    hf = np.fft.rfft(strain)
    white_hf = hf / (np.sqrt(interp_psd(freqs) /dt/2.))
    white_ht = np.fft.irfft(white_hf, n=Nt)
    return white_ht

# now whiten the data from H1 and L1, and also the NR template:
strain_H1_whiten = whiten(strain_H1,psd_H1,dt)
strain_L1_whiten = whiten(strain_L1,psd_L1,dt)



# We need to suppress the high frequencies with some bandpassing:
bb, ab = butter(4, [fband[0]*2./fs, fband[1]*2./fs], btype='band')
normalization = np.sqrt((fband[1]-fband[0])/(fs/2))
strain_H1_whitenbp = filtfilt(bb, ab, strain_H1_whiten) / normalization
strain_L1_whitenbp = filtfilt(bb, ab, strain_L1_whiten) / normalization


def plot_whiten():
    plt.figure()
    plt.plot(time-tevent, strain_H1_whitenbp, 'r', label='H1 strain', lw=1.5, alpha=0.8)
    plt.plot(time-tevent, strain_L1_whitenbp, 'g', label='L1 strain', lw=1.5, alpha=0.8)
    plt.xlim([3,3.1])
    plt.ylim([-3,3])
    plt.grid(True)
    plt.xlabel('time (s) since '+str(tevent))
    plt.ylabel('whitented strain')
    plt.legend(loc='best')
    plt.title( name +' Whitened')
    plt.savefig("Strain Figs/" + name +'_Whitened.'+plottype)
    
plot_strain()
# =============================================================================
# plot_stretched_strain()
# plot_H1_rolling_std()
# plot_L1_rolling_std()
# plot_ASD()
# =============================================================================
plot_whiten()