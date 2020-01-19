import numpy as np
import matplotlib.pyplot as plt
import readligo as rl
#----------------------------------------------------------------
# Load all GWOSC data from a single file 
#----------------------------------------------------------------
strain, time, chan_dict = rl.loaddata(
                          'H-H1_LOSC_4_V1-815411200-4096.hdf5', 'H1')
slice_list = rl.dq_channel_to_seglist(chan_dict['DATA'])
for slice in slice_list:
    time_seg = time[slice]
    strain_seg = strain[slice]
    # -- Do stuff with strain segment here