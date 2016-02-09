import numpy
import scipy
import pyfits

data = pyfits.open('/home/deen/Data/GRAVITY/SPARTA_Data/Casey/Matrices/DM2Z.fits')

newData = data[0].data[0:50,:]

HDU = pyfits.PrimaryHDU(newData)
HDU.writeto('LoopMonitor.DMPOS2MODES.fits', clobber=True)
