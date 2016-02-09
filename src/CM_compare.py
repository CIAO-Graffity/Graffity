import scipy
import numpy
import pyfits
import matplotlib.pyplot as pyplot

fig = pyplot.figure(0)

fig.clear()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

datadir = '/home/deen/Data/GRAVITY/MATLAB_DATA/2015-11-27/AssortedMatrices/CM_Computation/'

m1 = pyfits.getdata(datadir+'Recn.REC1.CM_1_cov_CMDLine.fits')
m2 = pyfits.getdata(datadir+'Recn.REC1.CM_3_cov_template.fits')
#m1 = pyfits.getdata(datadir+'Recn.REC1.CM_nominal_CMDLine.fits')
#m2 = pyfits.getdata(datadir+'Recn.REC1.CM_nominal_template.fits')

#m1 = pyfits.getdata(datadir+'SubApCov_a.fits')
#m2 = pyfits.getdata(datadir+'SubApCov_b.fits')

m3 = m1-m2

blah = ax.matshow(m3)

fig.colorbar(blah)

fig.show()

