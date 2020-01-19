import os, json, h5py
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, iirdesign, zpk2tf, freqz
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import readligo as rl


# if os.system('ls readligo.py') != 0:
#     os.system('wget https://www.gw-openscience.org/static/sample_code/readligo.py')


#eventname = 'GW170817'
num = input('Enter 1 for GW150914\n      2 for LVT151012\n      3 for GW151226\n      4 for GW170104\n')
eventname = ['GW150914', 'LVT151012', 'GW151226', 'GW170104'][num-1]


os.chdir('LOSC_Event_tutorial')

# Read the event properties from a local json file
fnjson = "BBH_events_v3.json"
try:
    events = json.load(open(fnjson,"r"))
except IOError:
    print("Cannot find resource file "+fnjson)
    print("You can download it from https://losc.ligo.org/s/events/"+fnjson)
    print("Quitting.")
    quit()

# did the user select the eventname ?
try: 
    events[eventname]
except:
    print('You must select an event that is in '+fnjson+'! Quitting.')
    quit()


# Extract the parameters for the desired event:
event = events[eventname]
fn_H1 = event['fn_H1']              # File name for H1 data
fn_L1 = event['fn_L1']              # File name for L1 data
fn_template = event['fn_template']  # File name for template waveform
fs = event['fs']                    # Set sampling rate
tevent = event['tevent']            # Set approximate event GPS time
fband = event['fband']              # frequency band for bandpassing signal
print("Reading in parameters for event " + event["name"])
print(event)


#filelist = ['https://dcc.ligo.org/public/0146/P1700337/001/H-H1_LOSC_C00_4_V1-1187006834-4096.hdf5',
#            'https://dcc.ligo.org/public/0146/P1700337/001/L-L1_LOSC_C00_4_V1-1187006834-4096.hdf5',
#            'https://dcc.ligo.org/public/0146/P1700337/001/H-H1_LOSC_C00_16_V1-1187006834-4096.hdf5',
#            'https://dcc.ligo.org/public/0146/P1700337/001/L-L1_LOSC_C00_16_V1-1187006834-4096.hdf5',
#            'https://dcc.ligo.org/public/0146/P1700349/001/H-H1_LOSC_CLN_4_V1-1187007040-2048.hdf5',
#            'https://dcc.ligo.org/public/0146/P1700349/001/L-L1_LOSC_CLN_4_V1-1187007040-2048.hdf5',

#filelist = ['https://dcc.ligo.org/public/0146/P1700349/001/H-H1_LOSC_CLN_16_V1-1187007040-2048.hdf5',
#            'https://dcc.ligo.org/public/0146/P1700349/001/L-L1_LOSC_CLN_16_V1-1187007040-2048.hdf5']

#for file in filelist:
#    if os.system('ls '+file[46:]) != 0:
#        os.system('wget '+file)


plottype = 'pdf'   #'png'


# (deprecated; from https://www.gw-openscience.org/s/events/GW150914/GW150914_tutorial.html)
#
# # Load data from H1
# fn_H1 = 'H-H1_LOSC_CLN_4_V1-1187007040-2048.hdf5'
# strain_H1, time_H1, chan_dict_H1 = rl.loaddata(fn_H1, 'H1')
#
# # and then from L1
# fn_L1 = 'L-L1_LOSC_CLN_4_V1-1187007040-2048.hdf5'
# strain_L1, time_L1, chan_dict_L1 = rl.loaddata(fn_L1, 'L1')
#
# # sampling rate:
# fs = 4096


#----------------------------------------------------------------
# Load LIGO data from a single file.
# FIRST, define the filenames fn_H1 and fn_L1, above.
#----------------------------------------------------------------
try:
    # read in data from H1 and L1, if available:
    strain_H1, time_H1, chan_dict_H1 = rl.loaddata(fn_H1, 'H1')
    strain_L1, time_L1, chan_dict_L1 = rl.loaddata(fn_L1, 'L1')
except:
    print("Cannot find data files!")
    print("You can download them from https://losc.ligo.org/s/events/"+eventname)
    print("Quitting.")
    quit()


# both H1 and L1 will have the same time vector, so:
time = time_H1
# the time sample interval (uniformly sampled!)
dt = time[1] - time[0]


# First, let's look at the data and print out some stuff:

# this doesn't seem to work for scientific notation:
# np.set_printoptions(precision=4)

print ('  time_H1: len, min, mean, max = ', \
   len(time_H1), time_H1.min(), time_H1.mean(), time_H1.max())
print ('strain_H1: len, min, mean, max = ', \
   len(strain_H1), strain_H1.min(),strain_H1.mean(),strain_H1.max())
print ('strain_L1: len, min, mean, max = ', \
   len(strain_L1), strain_L1.min(),strain_L1.mean(),strain_L1.max())
    
#What's in chan_dict? See https://losc.ligo.org/archive/dataset/GW150914/
bits = chan_dict_H1['DATA']
print ('H1     DATA: len, min, mean, max = ', len(bits), bits.min(),bits.mean(),bits.max())
bits = chan_dict_H1['CBC_CAT1']
print ('H1 CBC_CAT1: len, min, mean, max = ', len(bits), bits.min(),bits.mean(),bits.max())
bits = chan_dict_H1['CBC_CAT2']
print ('H1 CBC_CAT2: len, min, mean, max = ', len(bits), bits.min(),bits.mean(),bits.max())
bits = chan_dict_L1['DATA']
print ('L1     DATA: len, min, mean, max = ', len(bits), bits.min(),bits.mean(),bits.max())
bits = chan_dict_L1['CBC_CAT1']
print ('L1 CBC_CAT1: len, min, mean, max = ', len(bits), bits.min(),bits.mean(),bits.max())
bits = chan_dict_L1['CBC_CAT2']
print ('L1 CBC_CAT2: len, min, mean, max = ', len(bits), bits.min(),bits.mean(),bits.max())
print ('In both H1 and L1, all 32 seconds of data are present (DATA=1), ')
print ('and all pass data quality (CBC_CAT1=1 and CBC_CAT2=1).')



plt.rcParams['figure.figsize'] = 7.5, 4.5
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


if os.system('ls ../'+eventname) != 0:
    os.mkdir('../'+eventname)
os.chdir('../'+eventname)


# # GW170817 occurred at GPS time 1187008882.43
# assert tevent == 1187008882.43     # August 17 2017, 12:41:04.43 UTC

deltat = 5.                     # seconds around the event
# index into the strain time series for this time interval:
indxt = np.where((time_H1 >= tevent-deltat) & (time_H1 < tevent+deltat))
print('tevent = %.2f' % tevent)

plt.figure()
plt.plot(time_H1[indxt]-tevent, strain_H1[indxt], 'r', label='H1 strain', lw=1.2, alpha=0.6)
plt.plot(time_L1[indxt]-tevent, strain_L1[indxt], 'g', label='L1 strain', lw=1.2, alpha=0.6)
plt.grid('on')
plt.xlabel('time (s) since '+str(tevent))
plt.ylabel('strain')
plt.legend(loc='best')
plt.title('Advanced LIGO strain data near '+eventname)
plt.savefig(eventname+'_strain.'+plottype)



# number of sample for the fast fourier transform:
NFFT = 4*fs
f_min = 10
f_max = 2000
Pxx_H1, freqs = mlab.psd(strain_H1, Fs = fs, NFFT = NFFT)
Pxx_L1, freqs = mlab.psd(strain_L1, Fs = fs, NFFT = NFFT)

# We will use interpolations of the ASDs computed above for whitening:
psd_H1 = interp1d(freqs, Pxx_H1)
psd_L1 = interp1d(freqs, Pxx_L1)


# plot the ASDs:
plt.figure()
plt.loglog(freqs, np.sqrt(Pxx_H1), 'r', label='H1 strain', lw=1.2, alpha=0.6)
plt.loglog(freqs, np.sqrt(Pxx_L1), 'g', label='L1 strain', lw=1.2, alpha=0.6)
plt.axis([f_min, f_max, 1e-24, 1e-19])
plt.grid('on')
plt.ylabel('ASD (strain/rtHz)')
plt.xlabel('Freq (Hz)')
plt.legend(loc='best')
plt.title('Advanced LIGO strain data near '+eventname)
plt.savefig(eventname+'_ASDs.'+plottype)




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


st
# We need to suppress the high frequencies with some bandpassing:
bb, ab = butter(4, [fband[0]*2./fs, fband[1]*2./fs], btype='band')
normalization = np.sqrt((fband[1]-fband[0])/(fs/2))
strain_H1_whitenbp = filtfilt(bb, ab, strain_H1_whiten) / normalization
strain_L1_whitenbp = filtfilt(bb, ab, strain_L1_whiten) / normalization



plt.figure()
plt.plot(time-tevent, strain_H1_whitenbp, 'r', label='H1 strain', lw=1.5, alpha=0.8)
plt.plot(time-tevent, strain_L1_whitenbp, 'g', label='L1 strain', lw=1.5, alpha=0.8)
plt.xlim([-0.1,0.05])
plt.ylim([-10,10])
plt.grid('on')
plt.xlabel('time (s) since '+str(tevent))
plt.ylabel('whitented strain')
plt.legend(loc='best')
plt.title('Advanced LIGO WHITENED strain data near '+eventname)
plt.show()





