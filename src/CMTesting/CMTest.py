import Graffity
import pyfits
import matplotlib.pyplot as pyplot
import numpy

fig = pyplot.figure(0)
fig.clear()
ax1 = fig.add_axes([0.1, 0.1, 0.4, 0.4])
ax2 = fig.add_axes([0.1, 0.5, 0.4, 0.4])
ax3 = fig.add_axes([0.5, 0.1, 0.4, 0.4])
ax4 = fig.add_axes([0.5, 0.5, 0.4, 0.4])

matrix_dir = '/home/deen/Data/GRAVITY/MATLAB_DATA/2015-11-18/CM_testing/'
datadir='/home/deen/Data/GRAVITY/MATLAB_DATA/2015-11-18/DATA_LOGGER-130129/'

df = datadir+'CIAO_LOOP_0001.fits'

cb = Graffity.CircularBuffer(df)

HOCtr = Graffity.Controller(CM = matrix_dir+'CM_2.fits', iTT2HO=matrix_dir+'ITT2HO_2.fits', TT2HO=matrix_dir+'TT2HO.fits')

LA = Graffity.LoopAnalyzer(HOCtr=HOCtr, CB=cb)

LA.predict()

difference = cb.HODM[1:] - LA.predictions[:1]
ax1.matshow(numpy.array(LA.predictions).T, aspect='auto')
ax2.matshow(cb.HODM.T, aspect='auto')
blah = ax3.matshow(difference.T, aspect='auto')

fig.colorbar(blah)
fig.show()
