import numpy
import scipy
import pyfits
import os
from scipy.misc import factorial as fac
import scipy.interpolate as interp
import scipy.fftpack as fftpack
import matplotlib.pyplot as pyplot
from scipy import optimize
from scipy.ndimage import rotate
from PIL import Image
from scipy import signal

class CircularBuffer( object ):
    def __init__(self, df):
        self.df = df
        self.header = pyfits.getheader(df)
        self.data = pyfits.getdata(df)
        self.columns = self.data.columns
        self.Intensities = self.data.field('Intensities')
        self.Gradients = self.data.field('Gradients')
        self.HODM = self.data.field('HODM_Positions')
        self.TTM = self.data.field('ITTM_Positions')

class Controller( object ):
    def __init__(self, CM = None, iTT2HO=None, TT2HO=None):
        self.CM = pyfits.getdata(CM)
        self.HOCM = self.CM[:60,:]
        self.TTCM = self.CM[60:,:]
        self.iTT2HO = pyfits.getdata(iTT2HO)
        self.TT2HO = pyfits.getdata(TT2HO)

    def computeDeltas(self, slopes):
        return self.HOCM.dot(slopes)

class LoopAnalyzer( object):
    def __init__(self, HOCtr=None, CB =None):
        self.HOCtr = HOCtr
        self.CB = CB
        self.predictions = []

    def predict(self):
        for frame, mirror in zip(self.CB.Gradients, self.CB.HODM):
            self.predictions.append(mirror-self.HOCtr.computeDeltas(frame))

class FLIRCamImage( object ):
    def __init__(self, df):
        self.df = df
        self.image = Image.open(self.df)
        self.imdata = numpy.array(self.image)[:,:,:3].sum(axis=2)
        self.mean = numpy.mean(self.imdata[self.imdata > 0])
        self.max = numpy.max(self.imdata)
        self.imdata = self.imdata/float(self.max)
        #self.imdata /= numpy.mean(self.imdata[self.imdata > 0])
        self.floor = numpy.min(self.imdata[self.imdata > 0])
        self.nx = len(self.imdata[0])
        self.ny = len(self.imdata)
        self.npix = self.nx*self.ny

    def Spot(self, p):
        sig = p[0]
        xc = p[1]
        yc = p[2]

        if (p[0] <= 0.0):
            return numpy.zeros(self.ncutout)
        if (p[1] < 0) or (p[1] > self.cutout_x):
            return numpy.zeros(self.ncutout)
        if (p[2] < 0) or (p[2] > self.cutout_y):
            return numpy.zeros(self.ncutout)
        x = numpy.arange(self.cutout_x)
        y = numpy.arange(self.cutout_y)
        image = numpy.zeros((self.cutout_x, self.cutout_x))

        for i in range(self.cutout_x):
            for j in range(self.cutout_y):
                image[i][j] = 1.0*( numpy.exp(-(x[i]-xc)**2.0/sig)*
                        numpy.exp(-(y[j]-yc)**2.0/sig))

        maxval = numpy.max(image)
        image = image/maxval
        image[image < self.floor] = 0.0
        return image.ravel()

    def findFocusCenter(self, x, y, ax=None):
        """
        cutout should be 8x8 grid of pixels
        """

        cutout = self.extractCutout(x, y, 10)

        if ax!=None:
            ax.matshow(cutout)
            ax.figure.show()
            raw_input()
        self.cutout_x = len(cutout[0])
        self.cutout_y = len(cutout)
        self.ncutout = self.cutout_x*self.cutout_y

        sig = 2.4
        xcenter = self.cutout_x/2.0
        ycenter = self.cutout_y/2.0

        errfunc = lambda p, y : numpy.abs(self.Spot(p) - y)
        coeffs = [sig, xcenter, ycenter]
        pfit, success = optimize.leastsq(errfunc, coeffs, args=(cutout.ravel()))

        xfit = pfit[2] - xcenter + x
        yfit = pfit[1] - ycenter + y
        sigma = pfit[0]**0.5
        return sigma, xfit, yfit, cutout

    def findPupilCenter(self, x, y, zoomFactor, pupilImage=None, ax=None):
        """
        cutout should be 8x8 grid of pixels
        """

        cutout = self.extractCutout(x, y, 80, chopTop=True)

        cutoutZoom = self.zoomIn(cutout, zoomFactor)

        corr = signal.correlate2d(cutoutZoom, pupilImage, boundary='fill', mode='valid')
        xc, yc = numpy.unravel_index(numpy.argmax(corr), corr.shape)
        xzp = corr.shape[0]/2.0
        yzp = corr.shape[1]/2.0
        xcent = (xc-xzp)/zoomFactor + x
        ycent = (yc-yzp)/zoomFactor + y

        if ax!=None:
            ax.clear()
            ax.matshow(corr)
            ax.figure.show()
            raw_input()
        return xcent, ycent

    def extractCutout(self, x, y, size, chopTop=False):
        cutout = self.imdata[y-size:y+size, x-size:x+size].copy()

        if chopTop:
            cutoff = numpy.mean(cutout)
            cutout[cutout > cutoff] = 1.0
        return cutout

    def zoomIn(self, original, factor):
        x = numpy.arange(original.shape[0])
        y = numpy.arange(original.shape[1])
        interpolator = scipy.interpolate.RectBivariateSpline(x, y, original)
        new_x = numpy.linspace(0, x[-1], num=len(x)*factor)
        new_y = numpy.linspace(0, y[-1], num=len(y)*factor)
        xx, yy = numpy.meshgrid(new_x, new_y)
        zoomed = interpolator.ev(xx, yy).reshape(len(x)*factor, len(y)*factor)
        return zoomed

class NGCImage( object):
    def __init__(self, filename):
        self.filename = filename
        self.header = pyfits.getheader(filename)
        self.data = pyfits.getdata(filename)

    def subtract_background(self, background):
        self.subtracted = self.data - background.data

    def generate_model(self, n):
        feature_width = int(n*self.spacing+4*self.fwhm)
        feature_x = numpy.arange(feature_width)
        feature_y = numpy.zeros(feature_width)

        for i in range(n):
            c = (i+1)*self.spacing
            feature_y += self.height*numpy.exp(-(feature_x-c)**2.0/(2.0*self.fwhm/2.4)**2.0)

        return feature_x, feature_y

    def find_centers(self, n, collapse):
        distance, height = self.generate_model(n)
        y_corr = scipy.correlate(collapse, height)
        x_corr = scipy.linspace(0, len(y_corr)-1, num =len(y_corr))

        peak = x_corr[numpy.argsort(y_corr)[-1]]

        centers = []
        for i in range(n):
            centers.append((i+1)*self.spacing+peak)

        return numpy.array(centers)

    def findSubapertureCenters(self, nx=20, ny=21):
        self.nx = nx
        self.ny = ny
        self.xcollapse = self.subtracted.sum(axis=0)
        self.ycollapse = self.subtracted.sum(axis=1)

        self.fwhm = 2.5
        self.spacing = 8.1
        self.height = 0.75*numpy.max(self.xcollapse)

        self.xcenters = self.find_centers(nx, self.xcollapse)
        self.ycenters = self.find_centers(ny, self.ycollapse)

    def TopHat(self, p):
        sig_x = p[0]
        sig_y = p[1]
        theta = p[2]
        amplitude = numpy.deg2rad(p[3])
        xc = p[4]
        yc = p[5]

        if (sig_x < 0):# or (sig_x > 18.0):
            return numpy.zeros(64)
        if (sig_y < 0):# or (sig_y > 18.0):
            return numpy.zeros(64)
        fine_grid = numpy.linspace(0, 8)
        fine_image = numpy.zeros((len(fine_grid), len(fine_grid)))

        a = numpy.cos(theta)**2/(2.0*sig_x**2.0) + numpy.sin(theta)**2./(2.0*sig_y**2.0)
        b = -numpy.sin(2.0*theta)**2/(4.0*sig_x**2.0) + numpy.sin(2.0*theta)**2./(4.0*sig_y**2.0)
        c = numpy.sin(theta)**2/(2.0*sig_x**2.0) + numpy.cos(theta)**2./(2.0*sig_y**2.0)

        indices = range(len(fine_grid))
        
        for x in indices:
            for y in indices:
                fine_image[x][y] = amplitude*numpy.exp(-(a*(fine_grid[x]-xc)**2.0 +
                    2.0*b*(fine_grid[x]-xc)*(fine_grid[y]-yc) +
                    c*(fine_grid[y]-yc)**2.0))

        coarse_image = numpy.zeros((8,8))
        for x in range(8):
            for y in range(8):
                goodx = numpy.arange(len(fine_grid))[(x <= fine_grid) & (fine_grid < x+1)]
                goody = numpy.arange(len(fine_grid))[(y <= fine_grid) & (fine_grid < y+1)]
                coarse_image[x][y] = numpy.sum(fine_image[goodx][:,goody])

        return coarse_image.ravel()

    def Elipse(self, p):
        sig_x = p[0]
        sig_y = sig_x*p[1]
        theta = numpy.deg2rad(p[2])
        xc = p[3]
        yc = p[4]

        #print sig_x
        #print numpy.rad2deg(theta)
        #raw_input()
        if (p[1] >= 1.0):
            return numpy.zeros(64)
        if (p[2] < -180.0) or (p[2] > 180.0):
            return numpy.zeros(64)
        if (sig_x < 0):# or (sig_x > 18.0):
            return numpy.zeros(64)
        if (sig_y < 0):# or (sig_y > 18.0):
            return numpy.zeros(64)
        fine_grid = numpy.linspace(0, 8)
        fine_image = numpy.zeros((len(fine_grid), len(fine_grid)))

        a = numpy.cos(theta)**2/(2.0*sig_x**2.0) + numpy.sin(theta)**2./(2.0*sig_y**2.0)
        b = -numpy.sin(2.0*theta)**2/(4.0*sig_x**2.0) + numpy.sin(2.0*theta)**2./(4.0*sig_y**2.0)
        c = numpy.sin(theta)**2/(2.0*sig_x**2.0) + numpy.cos(theta)**2./(2.0*sig_y**2.0)

        indices = range(len(fine_grid))
        
        for x in indices:
            for y in indices:
                fine_image[x][y] = numpy.exp(-(a*(fine_grid[x]-xc)**2.0 +
                    2.0*b*(fine_grid[x]-xc)*(fine_grid[y]-yc) +
                    c*(fine_grid[y]-yc)**2.0))

        coarse_image = numpy.zeros((8,8))
        for x in range(8):
            for y in range(8):
                goodx = numpy.arange(len(fine_grid))[(x <= fine_grid) & (fine_grid < x+1)]
                goody = numpy.arange(len(fine_grid))[(y <= fine_grid) & (fine_grid < y+1)]
                coarse_image[x][y] = numpy.sum(fine_image[goodx][:,goody])

        cutoff = numpy.median(coarse_image) + numpy.std(coarse_image)
        bright = coarse_image > cutoff
        dark = coarse_image <= cutoff
        coarse_image[bright] = 1.0
        coarse_image[dark] /= cutoff
        return coarse_image.ravel()

    def fitElipse(self, cutout):
        """
        cutout should be 8x8 grid of pixels
        """
        sig_x = 2.4
        e = 0.8
        angle = 10.0
        xcenter = 3.5
        ycenter = 3.5

        errfunc = lambda p, y : numpy.abs(self.Elipse(p) - y)
        coeffs = [sig_x, e, angle, xcenter, ycenter]
        pfit = optimize.leastsq(errfunc, coeffs, args=(cutout))

        return pfit

    def extractCutout(self, x, y):
        xcenter = int(numpy.round(self.xcenters[x]))
        ycenter = int(numpy.round(self.ycenters[y]))

        cutout = self.subtracted[ycenter-4:ycenter+4, xcenter-4:xcenter+4]

        return cutout.copy()

    def twirl(self, ax1=None, ax2=None):
        xp = numpy.arange(0, 8, 1.0)
        yp = numpy.arange(0, 8, 1.0)
        grid_x = numpy.linspace(0, 7)
        grid_y = numpy.linspace(0, 7)
        grid = numpy.meshgrid(grid_x, grid_y)
        angles = numpy.arange(0, 360, 45.0)
        for i in range(self.nx):
            for j in range(self.ny):
                x = self.xcenters[i]
                y = self.ycenters[j]
                cutout = self.extractCutout(i, j)
                if not(ax1 == None):
                    ax1.clear()
                    ax2.matshow(cutout)
                    ax1.figure.show()
                if numpy.sum(cutout) > 400:
                    f = scipy.interpolate.interp2d(xp, yp, cutout, kind='cubic')
                    for theta in angles:
                        rotated = rotate(grid, theta)
                        print("%s" % rotated)
                        raw_input()


    def fitElipses(self, ax1=None, ax2=None, ax3=None):
        sig_x = []
        sig_y = []
        angle = []
        amplitude = []
        xc = []
        yc = []
        for i in range(self.nx):
            for j in range(self.ny):
                x = self.xcenters[i]
                y = self.ycenters[j]
                cutout = self.extractCutout(i, j)
                #print x, y, numpy.sum(cutout)
                if not(ax1 == None):
                    ax1.clear()
                    ax1.matshow(cutout, vmin = 0, vmax = 1)
                    ax1.figure.show()
                if numpy.sum(cutout) > 400:
                    cutoff = numpy.median(cutout)+ numpy.std(cutout)
                    good = cutout > cutoff
                    bad = cutout <= cutoff
                    cutout[good] = 1.0
                    cutout[bad] /= cutoff
                    fit, result = self.fitElipse(cutout.ravel())
                    if ax2 != None:
                        ax2.clear()
                        ax2.matshow(self.Elipse(fit).reshape(8,8), vmin = 0, vmax=1)
                        ax2.figure.show()
                        print("%s%" % fit)
                        raw_input()
                    sig_x.append(fit[0])
                    sig_y.append(fit[0]*fit[1])
                    angle.append(fit[2])
                    xc.append(fit[3]-3.5+x)
                    yc.append(fit[3]-3.5+y)
                    print("%.3f, %.3f" %( sig_y[-1], angle[-1]))
        self.sig_x = numpy.array(sig_x)
        self.sig_y = numpy.array(sig_y)
        self.angle = numpy.array(angle)
        self.amplitude = numpy.array(amplitude)
        self.xc = numpy.array(xc)

    def findCentroids(self, ax = None):
        xc = []
        yc = []
        for i in range(self.nx):
            for j in range(self.ny):
                x = self.xcenters[i]
                y = self.ycenters[j]
                cutout = self.extractCutout(i, j)
                if numpy.sum(cutout) > 400:
                    centroid = scipy.ndimage.measurements.center_of_mass(cutout)
                    xc.append(centroid[1] - len(cutout[0])/2.0+x)
                    yc.append(centroid[0] - len(cutout)/2.0+y)
                    if not(ax == None):
                        ax.clear()
                        ax.matshow(cutout)
                        ax.scatter(centroid[1], centroid[0], size=50)
                        ax.figure.show()
                        raw_input()

        self.xc = numpy.array(xc)
        self.yc = numpy.array(yc)

        #self.residual_x = self.xc % 1
        #self.residual_y = self.yc % 1

        #self.residual_x[self.residual_x > 0.5] -= 1.0
        #self.residual_y[self.residual_y > 0.5] -= 1.0


    def fit_line(self, x, y):
        fitfunc = lambda p, x : p[0]+(x*p[1])
        errfunc = lambda p, x, y: numpy.abs(fitfunc(p,x) - y)
        coeffs = [numpy.mean(y), 0.1]
        order = numpy.argsort(x)
        middle_x = numpy.array(x)[order[1:-1]]
        middle_y = numpy.array(y)[order[1:-1]]
        pfit = optimize.leastsq(errfunc, coeffs, args=(middle_x,middle_y) )

        #return numpy.arctan(numpy.abs(pfit[0][1]))*180.0/3.14159262
        return pfit


    def findAngles(self, ax = None):
        xangle = []
        residuals_x = []
        for x in self.xcenters:
            selected = numpy.abs(self.xc - x) < 2.0
            fit = self.fit_line(self.yc[selected], self.xc[selected])
            if not(ax == None):
                ax.plot(fit[0][0]+fit[0][1]*self.yc[selected], 
                        self.yc[selected], color = 'r')
            residuals_x.append(numpy.array([self.yc[selected], 
                fit[0][0]+fit[0][1]*self.yc[selected] - self.xc[selected]]))
            xangle.append(numpy.arctan(fit[0][1])*180.0/3.14159)

        yangle = []
        residuals_y = []
        for y in self.ycenters:
            selected = numpy.abs(self.yc - y) < 2.0
            fit = self.fit_line(self.xc[selected], self.yc[selected])
            if not(ax == None):
                ax.plot(self.xc[selected], fit[0][0]+fit[0][1]*self.xc[selected],
                        color = 'r')
            residuals_y.append(numpy.array([self.xc[selected], 
                fit[0][0]+fit[0][1]*self.xc[selected] - self.yc[selected]]))
            yangle.append(-numpy.arctan(fit[0][1])*180.0/3.14159)

        if not(ax==None):
            ax.scatter(self.xc, self.yc, s=10, c='r')

        self.xangle = xangle
        self.yangle = yangle
        self.residuals_x = residuals_x
        self.residuals_y = residuals_y

    



def twoDgaussian(x, y, center, stdev, A):
    retval = A * (numpy.exp(-(x-center[0])**2.0/stdev[0])*
                  numpy.exp(-(y-center[1])**2.0/stdev[1]))
    print("%.3f %.3f %.3f" % ( center, A, numpy.max(retval), numpy.max(x), numpy.min(x)))
    return retval

class zernikeMode(object):
    """
       Class representing a Zernike mode
    """
    def __init__(self, noll, mag):
        """
        input:  noll - Noll index
                mag - magnitude of Zernike mode - Units?
        """
        self.mag = mag
        if (noll == 2):
            self.n = 1
            self.m = 1
        elif (noll == 3):
            self.n = 1
            self.m = -1
        elif (noll == 4):
            self.n = 2
            self.m = 0
        elif (noll == 5):
            self.n = 2
            self.m = -2
        elif (noll == 6):
            self.n = 2
            self.m = 2
        else:
            self.n = 0
            self.m = 0

    def zernike_rad(self, rho):
        n=abs(self.n)
        m=abs(self.m)
        if (numpy.mod(n-m, 2) == 1):
            return rho*0.0
        
        wf = rho*0.0
        for k in range((n-m)/2+1):
            wf += rho**(n-2.0*k) * (-1.0)**k * fac(n-k) / ( fac(k) * fac( (n+m)/2.0 - k ) * fac( (n-m)/2.0 - k ) )
        
        return wf

    def setMag(self, mag):
        self.mag = mag

    def zernike(self, rho, phi, norm=False):
        nc = self.mag
        if (norm):
            nc = (2*(self.n+1)/(1+(self.m==0)))**0.5
        if (self.m > 0): return nc*self.zernike_rad(rho) * numpy.cos(self.m*phi)
        if (self.m < 0): return nc*self.zernike_rad(rho) * numpy.sin(-self.m*phi)
        return nc*self.zernike_rad(rho)

class pupil( object ):
    def __init__(self, x, y, innerRadius, outerRadius):
        self.x = x
        self.y = y
        self.inrad2 = innerRadius**2.0
        self.outrad2 = outerRadius**2.0

    def calculateFlux(self, x, y):
        pythag = (self.x-x)**2.0 + (self.y - y)**2.0
        if ( (pythag < self.outrad2) & (pythag > self.inrad2)):
            return 1.0
        else:
            return 0.0

    def calculateApertureIllumination(self, corner, size):
        area = 0.0
        illuminated = 0.0
        for x in numpy.linspace(corner[0], corner[0]+size, 50):
            for y in numpy.linspace(corner[1], corner[1]+size, 50):
                area += 1.0
                pythag = (self.x-x)**2.0 + (self.y - y)**2.0
                if ( (pythag < self.outrad2) & (pythag > self.inrad2)):
                    illuminated += 1.0
        return illuminated/area

    def getDecenter(self):
        return self.x, self.y

    def setDecenter(self, x, y):
        self.x = x
        self.y = y

class deformableMirror( object ):
    def __init__(self, parent):
        self.nActuators = 60
        self.influenceFunctions = pyfits.getdata(parent.datadir+'IF_cube.fits')
        nx = len(self.influenceFunctions[0][0])
        ny = len(self.influenceFunctions[0])
        xcoords = numpy.linspace(-888.0, 888.0, nx)
        ycoords = numpy.linspace(-888.0, 888.0, ny)
        self.interpFunctions = []
        self.actuatorPositions = numpy.zeros(len(self.influenceFunctions))
        for inflfunc in self.influenceFunctions:
            self.interpFunctions.append(interp.interp2d(xcoords, ycoords,
                            inflfunc, kind='cubic'))
        
    def setMirror(self, actPos):
        self.actuatorPositions = actPos

    def calcPosition(self, x, y):
        retval = 0.0
        for IF, amp in zip(self.interpFunctions, self.actuatorPositions):
            retval += amp*IF(x, y)[0]
        return retval

class derotator( object ):
    def __init__(self, parent):
        self.angle = 0.0
        self.parent = parent

    def setAngle(self, angle):
        self.angle = angle
        self.sine = numpy.sin(self.angle)
        self.cosine = numpy.cos(self.angle)

    def getMirrorPosition(self, x, y):
        newx = x*self.cosine-y*self.sine
        newy = x*self.sine+y*self.cosine
        return newx, newy

class WFS ( object ):
    def __init__(self, wavelength = 1800.0, beamSize = 1776.0, angle = 0.0):
        self.beamSize = beamSize
        self.wavelength = wavelength
        self.centObscScale = 1.116/8.00
        self.datadir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+'/data/'
        self.detector = detector(self, beamSize = beamSize)
        self.lenslet = lensletArray(self, angle = angle)
        self.wavefront = waveFront(self)
        self.pupil = pupil(0.0, 0.0, innerRadius=self.beamSize/2.0*self.centObscScale,
                outerRadius=self.beamSize/2.0)
        self.DM = deformableMirror(self)
        self.derotator = derotator(self)
        self.centroids = []

    def setupInstrument(self, zern, pupil, actuatorPokes, derotAngle, lensletAngle):
        """
        Generates an image seen by the detector of a wavefront described by
        the zernike coefficients in zern

        zern = [tip, tilt, defocus, astig1, astig2]
        pupil = [x, y]
        actuatorPokes = list of 60 actuator positions
        """
        self.derotator.setAngle(numpy.deg2rad(derotAngle))
        self.lenslet.setClockingAngle(lensletAngle)
        self.pupil.setDecenter(pupil[0], pupil[1])
        self.wavefront.setZern(zern)
        self.DM.setMirror(actuatorPokes)

    def expose(self):
        self.centroids.append(self.detector.expose())

    def calcWaveFront(self, x, y):
        wave = self.wavefront.calcWaveFront(x, y)
        #rotatedPosition = self.derotator.getMirrorPosition(x, y)
        #mirror = self.DM.calcPosition(rotatedPosition[0], rotatedPosition[1])
        #return wave + mirror
        return wave

class detector( object ):
    """
        The Detector class allows us to simulate the detector in the cryostat.
        Some of the numbers are most likely wrong, but with tweaking we should
        be able to simulate the effects of simple Zernike aberations of the 
        wave front on the spots on the detector.
    """
    def __init__(self, parent, beamSize=1776.0):
        self.parent = parent
        # Calculates the relative size of the central obscuration (M2)
        self.scramblingMap = pyfits.getdata(
                            self.parent.datadir+"scramblemap.fits")
        self.unscramblingMap = pyfits.getdata(
                            self.parent.datadir+"unscramblemap.fits")
        self.windowMap = pyfits.getdata(
                            self.parent.datadir+"windowmap.fits")
        self.SLsubapmap = self.parent.datadir+"LoopDisplaySrv.SUBAP_MAP.fits"
        self.beamSize = beamSize
        self.pixPerSubAp = 8
        self.nx = 72
        self.ny = 72
        self.readoutMode = "8x8"
        self.spacing = 24.0 #microns  Is this value correct?
        self.xpix = (numpy.arange(self.nx)-self.nx/2.0)*self.spacing
        self.ypix = (numpy.arange(self.ny)-self.ny/2.0)*self.spacing
        self.stdev = (8.0*self.spacing, 8.0*self.spacing)
        self.z = []
        self.frames = []
        self.centroids = []
        #self.weightingMap = self.makeWeightingMap()

    def scrambleFrame(self):
        scrambledFrame = numpy.zeros(6912)
        flatframe = self.z[-1].ravel()
        for y, i in zip(flatframe, self.scramblingMap):
            scrambledFrame[i]=y

        self.frames.append(scrambledFrame)
        

    def makeRamp(self):
        z = numpy.zeros((self.ny, self.nx))
        k = 0
        for i in range(self.nx):
            for j in range(self.ny):
                z[i][j] = k
                k += 1
        self.z.append(z)
        self.scrambleFrame()
        self.centroids.append([])

    
    def expose(self):
        debug = False
        centroids = []
        subsamp = 30.0
        FWHM_i = 1.1 * subsamp # FWHM in Pixel space
        FWHM_k = 0.88*self.pixPerSubAp*subsamp/FWHM_i
        location = int(round((self.pixPerSubAp*subsamp - FWHM_k)/2.0))
        delta = self.parent.lenslet.spacing/2.0
        nPixPoints = self.pixPerSubAp*subsamp + 1
        z = numpy.zeros((self.ny, self.nx))
        ptsOnDetector = numpy.linspace(-nPixPoints/2.0*self.spacing, 
                nPixPoints*self.spacing/2.0, nPixPoints)
        gridx, gridy = numpy.meshgrid(ptsOnDetector, ptsOnDetector)
        gridx /= subsamp
        gridy /= subsamp
        totalFlux = 0.0
        if debug:
            fig = pyplot.figure(0)
            fig.clear()
            ax1 = fig.add_axes([0.1, 0.1, 0.4, 0.4])
            ax2 = fig.add_axes([0.1, 0.5, 0.4, 0.4])
            ax3 = fig.add_axes([0.5, 0.1, 0.4, 0.4])
            ax4 = fig.add_axes([0.5, 0.5, 0.4, 0.4])
            extract = []
            count = 1
        for coord in self.parent.lenslet.coordinates:
            flux = numpy.zeros((nPixPoints, nPixPoints), dtype=numpy.complex)
            #intensity = numpy.zeros((nPixPoints, nPixPoints))
            #phase = numpy.zeros((nPixPoints, nPixPoints))
            i = location
            for x in numpy.linspace(coord[0]-delta, coord[0]+delta, 
                    num=int(round(FWHM_k))):
                j = location
                for y in numpy.linspace(coord[1]-delta, coord[1]+delta, 
                        num=int(round(FWHM_k))):
                    flux[j][i] = numpy.complex(self.parent.pupil.calculateFlux(x, y),
                            self.parent.calcWaveFront(x, y))
                    #intensity[i][j] = self.pupil.calculateFlux(x, y)
                    #phase[i][j] = self.calcWaveFront(x, y)
                    totalFlux += 1
                    j += 1
                i += 1
            flux[flux.real == 0.0] = numpy.complex(0.0, 0.0)
            sig = abs(flux) > 0
            flux[sig] -= numpy.complex(0, numpy.mean(flux.imag[sig]))
            image = fftpack.fft2(flux)
            image = fftpack.fftshift((image*image.conjugate()).real)
            if debug:
                if count in [8, 10, 15]:
                    extract.append(image)
                    count += 1
            xc = coord[0]+gridx
            yc = coord[1]+gridy
            inviewx = (self.xpix > numpy.min(xc)) & (self.xpix <= numpy.max(xc))
            inviewy = (self.ypix > numpy.min(yc)) & (self.ypix <= numpy.max(yc))
            weightX = 0.0
            weightY = 0.0
            denom = 0.0
            for i in range(self.nx):
                if inviewx[i]:
                    for j in range(self.ny):
                        if inviewy[j]:
                            fp = scipy.where( (xc >= self.xpix[i]) & (xc < self.xpix[i]+self.spacing+0.5) & (yc >= self.ypix[j]) & (yc < self.ypix[j]+self.spacing+0.5))
                            z[j][i] = numpy.sum(image[fp])
                            weightX += z[j][i]*i
                            weightY += z[j][i]*j
                            denom += z[j][i]
                            #print i, j, z[i][j], coord
                    #raw_input()
            centroids.append([weightX/denom, weightY/denom])

        if debug:
            ax1.matshow(extract[0])
            ax2.matshow(extract[1])
            ax3.matshow(extract[2])
            ax4.matshow(extract[2]-extract[0])
            fig.show()
            print("%.3f" % numpy.max(extract[0]))
            print("%.3f" % numpy.max(extract[1]))
            print("%.3f" % numpy.max(extract[2]))
            print("%.3f" % numpy.max(extract[2]-extract[0]))
            input()
        self.z.append(z)
        self.centroids.append(numpy.array(centroids))
        self.scrambleFrame()

        return self.centroids[-1]


    def calculateCentroids(self, zern, actuatorPokes):
        """
            Calcualates the locations of the centroids under the given 
            Zernike coefficients
        """
        self.wavefront.setZern(zern)
        self.DM.setMirror(actuatorPokes)
        dx = 10.0   # Microns
        dy = 10.0   # Microns
        centroids = []
        intensities = []
        DCx, DCy = self.pupil.getDecenter()  # Decenter of Pupil
        for c in self.lenslet.coordinates:
            # Calculates the partial derivatives
            zxp = self.calcWaveFront(c[0]+DCx+dx, c[1]+DCy)
            zxm = self.calcWaveFront(c[0]+DCx-dx, c[1]+DCy)
            zyp = self.calcWaveFront(c[0]+DCx, c[1]+DCy+dy)
            zym = self.calcWaveFront(c[0]+DCx, c[1]+DCy-dy)
            delx = (zxp - zxm)/(2)
            dely = (zyp - zym)/(2)

            # Computes the normal vector to the surface
            normalx = -delx*dy
            normaly = -dely*dx
            normalz = dx*dy

            #Calculates the shift in microns on the detector
            theta_x = scipy.arctan2(normalx, normalz)
            theta_y = scipy.arctan2(normaly, normalz)
            shift_x = scipy.tan(theta_x)*self.lenslet.fl
            shift_y = scipy.tan(theta_y)*self.lenslet.fl
            centroids.append([c[0]+shift_x, c[1]+shift_y])

            intensities.append(self.pupil.calculateApertureIllumination(
                               [c[0]-self.lenslet.spacing/2.0, 
                               c[1]-self.lenslet.spacing/2.0],
                               self.lenslet.spacing))

        return numpy.array(centroids), numpy.array(intensities)

    def makeWeightingMap(self):
        weightingMap = numpy.zeros((72,72),dtype=numpy.float32)

        for coord in self.lenslet.coords:
            x = coord[0]
            y = coord[1]
            print ("Not done yet!")
                    
        
    def calcWaveFront(self, x, y):
        wave = self.wavefront.calcWaveFront(x, y)
        #rotatedPosition = self.derotator.getMirrorPosition(x, y)
        #mirror = self.DM.calcPosition(rotatedPosition[0], rotatedPosition[1])
        #return wave + mirror
        return wave

    def saveRawFrames(self, filename):
        """
        Saves the raw (unscrambled) frames to a data file
        """

        self.z = numpy.array(self.z)
        hdu=pyfits.PrimaryHDU(self.z)
        hdu.writeto(filename, clobber=True)

    def saveFrames(self, filename):
        """
        Saves the frames to a SPARTA-readable data file
        """
        self.frames = numpy.array(self.frames)
        hdu = pyfits.PrimaryHDU(self.frames)
        hdu.scale('int16', bzero=32768, bscale=1)
        hdu.writeto(filename, clobber=True)

    def saveCentroids(self, filename):
        """
        Saves the calculated centroids to a fits file
        """
        self.centroids = numpy.array(self.centroids)
        hdu = pyfits.PrimaryHDU(self.centroids)
        hdu.writeto(filename, clobber=True)

class lensletArray( object ):
    """
    This class simulates the lenslet array
    """
    def __init__(self, parent=None, spacing=192.0, fl=2095.0, angle=0.0):
        """
            Spacing - spacing between adjacent lenslets (in microns)
            fl - focal length of individual lenslet (in microns)
        """
        if parent:
            self.parent = parent
            self.SLapertureMap = pyfits.getdata(self.parent.detector.SLsubapmap)
        
        else:
            self.SLapertureMap = [[False,False,True,True,True,True,True,False,False],
               [False, True, True, True, True, True, True, True, False],
               [True, True, True, True, True, True, True, True, True],
               [True, True, True, True, True, True, True, True, True],
               [True, True, True, True, False, True, True, True, True],
               [True, True, True, True, True, True, True, True, True],
               [True, True, True, True, True, True, True, True, True],
               [False, True, True, True, True, True, True, True, False],
               [False, False, True, True, True, True, True, False, False]]
        self.spacing = spacing   # Lenslet Array Spacing in microns
        self.fl = fl
        self.angle = numpy.deg2rad(angle)
        self.calculateCentroids()

    def setClockingAngle(self, angle):
        self.angle = numpy.deg2rad(angle)
        self.calculateCentroids()

    def calculateCentroids(self):
        coords = []

        for i in range(9):
            for j in range(9):
                if self.SLapertureMap[i][j]:
                    y = (i-4)*self.spacing
                    x = (j-4)*self.spacing
                    coords.append((x*numpy.cos(self.angle)-y*numpy.sin(self.angle),
                        x*numpy.sin(self.angle)+y*numpy.cos(self.angle)))

        self.coordinates = coords


class waveFront( object ):
    """
    This object describes the wavefront as it hits the lenslet array
    """
    def __init__(self, parent, beamSize = 1776.0):
        self.parent = parent
        self.beamSize = beamSize
        self.tip = zernikeMode(2, 0.00)
        self.tilt = zernikeMode(3, 0.00)
        self.defocus = zernikeMode(4, 0.00)
        self.astig1 = zernikeMode(5, 0.00)
        self.astig2 = zernikeMode(6, 0.00)
    
    def setZern(self, zern):
        """
            Sets the magnitudes of the Zernike components.
        """
        for mag, z in zip(zern,
                [self.tip, self.tilt, self.defocus, self.astig1, self.astig2]):
            z.setMag(mag*2.0*numpy.pi/self.parent.wavelength)

    def calcWaveFront(self, x, y):
        """
        Sums through the different Zernike components at a particular location
        on the wavefront (x, y) to find the local zernike magnitude.
        """
        rho = (x**2.0 + y**2.0)**(0.5)/(self.beamSize/2.0)
        phi = numpy.arctan2(y, x)
        value = 0.0
        for zern in [self.tip, self.tilt, self.defocus,self.astig1,self.astig2]:
            value += zern.zernike(rho, phi)
        return value

    def pokeActuator(self, x, y, InflFunc):
        """
            interpolates the value of an actuator poke at a certain x and y
        """
        return InflFunc(x, y)


