import numpy
import scipy
import pyfits
import os
from scipy.misc import factorial as fac
import scipy.interpolate as interp
import scipy.fftpack as fftpack
import matplotlib.pyplot as pyplot

def twoDgaussian(x, y, center, stdev, A):
    retval = A * (numpy.exp(-(x-center[0])**2.0/stdev[0])*
                  numpy.exp(-(y-center[1])**2.0/stdev[1]))
    print center, A, numpy.max(retval), numpy.max(x), numpy.min(x)
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
        self.datadir = os.path.dirname(os.path.dirname(__file__))+'/data/'
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
            print numpy.max(extract[0])
            print numpy.max(extract[1])
            print numpy.max(extract[2])
            print numpy.max(extract[2]-extract[0])
            raw_input()
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
            print "Not done yet!"
                    
        
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


