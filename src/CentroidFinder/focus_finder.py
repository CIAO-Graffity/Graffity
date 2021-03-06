import Graffity
import matplotlib.pyplot as pyplot
import numpy
import scipy
import glob
import PIL
import images2gif
import io

fig = pyplot.figure(0)
fig.clear()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

directory= '/home/fprakt/Data/Derotator/derot_10112015/'
files = glob.glob(directory+'focus*001.TIF')

x = []
y = []
sigma = []
angle = []
cutouts = []

for df in files:
    image = Graffity.FLIRCamImage(df)
    angle.append(float(df.split('/')[-1].split('_')[1].split('deg')[0]))
    blah = image.findFocusCenter(166, 87)
    sigma.append(blah[0])
    x.append(blah[1])
    y.append(blah[2])
    cutouts.append(blah[3]*255)
    #cutouts.append(PIL.Image.fromarray(numpy.uint8(blah[3]*255)))

x = numpy.array(x)
y = numpy.array(y)
angle = numpy.array(angle)
order = numpy.argsort(angle)
#ax.plot(x[order], y[order])

frames = []
buf = []
#outfile = open('Focus_Runout_10nov.txt', 'w')
outfile = open('focus_0.txt', 'w')
outfile.write("%5s %16s %16s\n" % ( 'angle', 'x' , 'y'))
outfile.write("%5s %16s %16s\n" % ( '-----', '-------------', '---------'))

for i in order:
    ax.clear()
    ax.matshow(cutouts[i])
    ax.plot(x[order]-166+10, y[order]-87+10, color = 'y', lw=4.0)
    ax.scatter(x[i]-166+10, y[i]-87+10, color = 'k', s=85.0)
    ax.set_xbound(lower=0, upper=19)
    ax.set_ybound(lower=0, upper=19)
    ax.text(1.0, 1.0, 'Angle = %d' % angle[i], fontsize=20, color = 'y')
    ax.text(1.0, 2.0, 'X = %.2f' % x[i], fontsize=20, color = 'y')
    ax.text(1.0, 3.0, 'Y = %.2f' % y[i], fontsize=20, color = 'y')
    buf.append(io.BytesIO())
    #fig.show()
    fig.savefig(buf[-1], format='png')
    buf[-1].seek(0)
    print("%5s %17s %17s" % ( '%.1f' % angle[i], '%.5f' % x[i], '%.5f' %y [i]))
    outfile.write("%5s %16s %16s\n" % ( '%.1f' % angle[i], '%.5f' % x[i], '%.5f' %y [i]))
    #outfile.write("%d %.2f %.2f\n" % (angle[i], x[i], y[i]))
    #raw_input()
    frames.append(PIL.Image.open(buf[-1]))

outfile.close()
images2gif.writeGif('Focus_Runout.gif', frames, duration=0.5)
#ax.matshow(image.imdata, vmin = 0.0, vmax=1.0)


fig.show()
