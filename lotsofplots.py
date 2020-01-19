import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import readligo as rl
fileName = 'V-V1_GWOSC_O2_4KHZ_R1-1185615872-4096.hdf5'
strain, time, channel_dict = rl.loaddata(fileName)
ts = time[1] - time[0] #-- Time between samples
fs = int(1.0 / ts)          #-- Sampling frequency
segList = rl.dq_channel_to_seglist(channel_dict['DEFAULT'], fs)
length = 16  # seconds
strain_seg = strain[segList[0]][0:(length*fs)]
time_seg = time[segList[0]][0:(length*fs)]
fig = plt.figure(figsize=(10,10))
fig.subplots_adjust(wspace=0.3, hspace=0.3)
plt.subplot(321)
plt.plot(time_seg - time_seg[0], strain_seg)
plt.xlabel('Time since GPS ' + str(time_seg[0]))
plt.ylabel('Strain')

window = np.blackman(strain_seg.size)
windowed_strain = strain_seg*window
freq_domain = np.fft.rfft(windowed_strain) / fs
freq = np.fft.rfftfreq(len(windowed_strain))*fs

plt.subplot(322)
plt.loglog( freq, abs(freq_domain) )
plt.axis([10, fs/2.0, 1e-24, 1e-18])
plt.grid('on')
plt.xlabel('Freq (Hz)')
plt.ylabel('Strain / Hz')

#----------------------------------
# Make PSD for first chunk of data
#----------------------------------
plt.subplot(323)
Pxx, freqs = mlab.psd(strain_seg, Fs = fs, NFFT=fs)
plt.loglog(freqs, Pxx)
plt.axis([10, 2000, 1e-46, 1e-36])
plt.grid('on')
plt.ylabel('PSD')
plt.xlabel('Freq (Hz)')

#-------------------------
# Plot the ASD
#-------------------------------
plt.subplot(324)
plt.loglog(freqs, np.sqrt(Pxx))
plt.axis([10, 2000, 1e-23, 1e-18])
plt.grid('on')
plt.xlabel('Freq (Hz)')
plt.ylabel('ASD [Strain / Hz$^{1/2}$]')

#--------------------
# Make a spectrogram
#-------------------
NFFT = 1024
window = np.blackman(NFFT)
plt.subplot(325)
spec_power, freqs, bins, im = plt.specgram(strain_seg, NFFT=NFFT, Fs=fs, 
                                    window=window)
plt.xlabel('Time (s)')
plt.ylabel('Freq (Hz)')

#------------------------------------------
# Renormalize by average power in freq. bin
#-----------------------------------------
med_power = np.zeros(freqs.shape)
norm_spec_power = np.zeros(spec_power.shape)
index = 0
for row in spec_power:
    med_power[index] = np.median(row)
    norm_spec_power[index] = row / med_power[index]
    index += 1

ax = plt.subplot(326)
ax.pcolormesh(bins, freqs, np.log10(norm_spec_power))
plt.xlabel('Time (s)')
plt.ylabel('Freq (Hz)')

plt.show()
