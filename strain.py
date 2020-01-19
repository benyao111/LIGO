#----------------------
# Import needed modules
#----------------------
import pandas
from pandas import rolling_std
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import readligo as rl

#---------------------
# Read in strain data
#---------------------
fileName1 = 'LIGO\H1_1185619968.hdf5'
strain1, time1, channel_dict1 = rl.loaddata(fileName1, 'H1')
ts1 = time1[1] - time1[0] #-- Time between samples
fs1 = int(1.0 / ts1)          #-- Sampling frequency

fileName2 = 'LIGO\L1_1185619968.hdf5'
strain2, time2, channel_dict2 = rl.loaddata(fileName2, 'H1')
ts2 = time2[1] - time2[0] #-- Time between samples
fs2 = int(1.0 / ts2)          #-- Sampling frequency

#---------------------------------------------------------
# Find a good segment, get first 16 seconds of data
#---------------------------------------------------------
segList1 = rl.dq_channel_to_seglist(channel_dict1['DEFAULT'], fs1)
length1 = 16  # seconds
strain_seg1 = strain1[segList1[0]][0:(length1*fs1)]
time_seg1 = time1[segList1[0]][0:(length1*fs1)]

segList2 = rl.dq_channel_to_seglist(channel_dict2['DEFAULT'], fs2)
length2 = 16  # seconds
strain_seg2 = strain2[segList2[0]][0:(length2*fs2)]
time_seg2 = time2[segList2[0]][0:(length2*fs2)]

#---------------------
# Plot the time series
#----------------------
plt.plot(time_seg1 - time_seg1[0], strain_seg1)
pandas.rolling_std(strain_seg1, 1) 
#plt.title('H1_1185619968 + L1_1185619968')
plt.ylim(-2e-18,2e-18)
plt.xlabel('Time since GPS ' + str(time_seg1[0]))
plt.ylabel('Strain')

plt.show()
#plt.savefig(C:/Users/Ben/Desktop/LIGO/1187729408)
