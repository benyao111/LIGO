import numpy as np 
import matplotlib.pyplot as plt
from astropy.visualization import make_lupton_rgb
from astropy.io import fits

image_r = fits.getdata('m101_uv.fits')
image_g = fits.getdata('m101_optical.fits')
image_b = fits.getdata('m101_xray.fits')

image_r1 = np.array(image_r, float)
image_g1 = np.array(image_g, float)
image_b1 = np.array(image_b, float)

image = make_lupton_rgb(image_r1, image_g1, image_b1, stretch=0.5)
plt.imshow(image)
plt.show()
