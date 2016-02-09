import scipy
import pyfits
import scipy.linalg

z2s = pyfits.getdata("../data/Z2S_136s_119z.fits")

s2z = scipy.linalg.pinv(z2s)

s2Zernike = pyfits.PrimaryHDU(data=s2z[:,:10])

s2Zernike.writeto("slopes2zerns.fits")

