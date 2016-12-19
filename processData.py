#!/usr/bin/python

#parse arguments
import argparse

parser = argparse.ArgumentParser(description='EEG data processor.')
parser.add_argument('--kmeans', dest='useKmeans', action='store_const',
                   const=True, default=False,
                   help='Use k-means to find centroids (default: use precalculated centroids)')
parser.add_argument('--fs',metavar='FREQUENCY', dest='fs', action='store',type = int, default=100,
                   help='data sampling frequency [Hz] (default: 100)'  )
parser.add_argument('--et',metavar='SECONDS', dest='et', action='store',type = int, default=30,
                   help='Epoch time, interval of result data points [s] (default: 30)'  )
parser.add_argument('filename', metavar='FILENAME',  help='a csv data filename')

args = parser.parse_args()

fs = args.fs	# data sampling frequency [Hz]
et = args.et	# epoch time [s]
sc = et*fs	# sample count per epoch
useKmeans = args.useKmeans

import sys
import os

if not os.path.exists(args.filename):
    sys.exit('ERROR: %s: no such file ' % args.filename)

import numpy as np
import math
from pandas import read_csv

df = read_csv(args.filename)
data = df.values

yRaw = data[:,1] # signal

ec = int(math.ceil(yRaw.size/sc)) # epochs count

spect = np.abs(np.fft.rfft(yRaw[0*sc:1*sc]))#first data chunk made to set dimenstion

#from signal import spectrogram
# unfortunately "spectrogram" is missing in my distribution I have to write it by myself.
#perform fft transformation on epoch data chunks

from scipy.fftpack import fft
for i in range(1, ec) :
	spec = np.fft.rfft(yRaw[sc*(i-1):sc*i])
	spect.resize(i+1, spec.size)
	spect[i,:] = np.abs(spec)

spect = np.abs(spect)

#aggregate data by frequency rangesd
freq = np.arange((sc/2)+1)/(float(et)) # frequency range
freqsize = freq.size

rangeBordersFrequencies = np.array([0.3,4,8,10.5,12.5,14,20])
rangeBorderIndexes = np.round(rangeBordersFrequencies*(sc+2)/fs)


aggregatedByFreqRanges = np.sum(spect[:,0:rangeBorderIndexes[0]], axis = 1)
for i in range(1, rangeBorderIndexes.size) :

	aggregatedByFreqRange = np.sum(spect[:,rangeBorderIndexes[i-1]:rangeBorderIndexes[i]], axis = 1)
	aggregatedByFreqRanges.resize(i+1,aggregatedByFreqRange.size)
	aggregatedByFreqRanges[i,:] = aggregatedByFreqRange

#normalize
aggNormalised = np.empty_like(aggregatedByFreqRanges);
for i in range(0, aggNormalised.shape[0]) :
	aggNormalised[i,:] = aggregatedByFreqRanges[i,:]/np.amax(aggregatedByFreqRanges[i,:])

#Split data into clusters
from scipy.cluster.vq import kmeans,vq

if useKmeans :
	clusterCenters, _ = kmeans(aggNormalised.T, 5)
	print clusterCenters
else :
	clusterCenters = read_csv('centers.csv').values

clusterTable, _ = vq(aggNormalised.T, clusterCenters)


#make PCA transformation (just for visualisation)
from matplotlib.mlab import PCA
spect_pca = PCA(aggNormalised.T).Y

#Processing steps presentation:
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

app = QtGui.QApplication([])

win1 = pg.GraphicsWindow(title="Data processing steps")
win1.resize(1800,700)

p1 = win1.addPlot(title="1. RawData", y=yRaw)

img_spect = pg.ImageItem()
img_spect.setImage(spect, autoLevels=True)
p2 = win1.addPlot(title="2. Spectrogram")
p2.addItem(img_spect)

win1.nextRow()

img_spect_agg = pg.ImageItem()
img_spect_agg.setImage(aggregatedByFreqRanges.T, autoLevels=True)
p3 = win1.addPlot(title="3. Aggregated by frequency ranges")
p3.addItem(img_spect_agg)

img_spect_agg_norm = pg.ImageItem()
img_spect_agg_norm.setImage(aggNormalised.T, autoLevels=True)
p4 = win1.addPlot(title="4. Normalized by frequency ranges")
p4.addItem(img_spect_agg_norm)


# Result presentation.

timeAxis = np.arange(ec)*et/60 #time of samples in minutes

win2 = pg.GraphicsWindow(title="Result")
win2.resize(1800,100)
p5a = win2.addPlot(title="A. clusterization shown on time graph", x = timeAxis)
p5a.plot(x = timeAxis[clusterTable==0], y = np.ones(ec)[clusterTable==0]*0, pen=None, symbol=1, symbolPen=None, symbolSize=5,  symbolBrush=(10, 100, 255, 70))
p5a.plot(x = timeAxis[clusterTable==1], y = np.ones(ec)[clusterTable==1]*0, pen=None, symbol=1, symbolPen=None, symbolSize=5,  symbolBrush=(200, 200, 200, 255))
p5a.plot(x = timeAxis[clusterTable==2], y = np.ones(ec)[clusterTable==2]*0, pen=None, symbol=1, symbolPen=None, symbolSize=5, symbolBrush=(160, 0, 10, 170))
p5a.plot(x = timeAxis[clusterTable==3], y = np.ones(ec)[clusterTable==3]*0, pen=None, symbol=1, symbolPen=None, symbolSize=5, symbolBrush=(100, 250, 90, 70))
p5a.plot(x = timeAxis[clusterTable==4], y = np.ones(ec)[clusterTable==4]*0, pen=None, symbol=1, symbolPen=None, symbolSize=5, symbolBrush=(200, 200, 10, 70))
p5a.showAxis('left', False)
p5a.setLabel('bottom', "Time", units='minutes')

win3 = pg.GraphicsWindow(title="Result")
win3.resize(700,700)
# this would be much more fun with 3 PCA components in 3d. Unfortunately 3d plots are not availabe on my crappy computer.
p5a = win3.addPlot(title="B. clusterization shown on 2 main PCA components")
p5a.plot(x= spect_pca[clusterTable==0,0], y = spect_pca[clusterTable==0,1], pen=None, symbol='o', symbolPen=None, symbolSize=7, symbolBrush=(10, 100, 255, 70))
p5a.plot(x= spect_pca[clusterTable==1,0], y = spect_pca[clusterTable==1,1], pen=None, symbol='o', symbolPen=None, symbolSize=7, symbolBrush=(200, 200, 200, 228))
p5a.plot(x= spect_pca[clusterTable==2,0], y = spect_pca[clusterTable==2,1], pen=None, symbol='o', symbolPen=None, symbolSize=7, symbolBrush=(160, 0, 10, 170))
p5a.plot(x= spect_pca[clusterTable==3,0], y = spect_pca[clusterTable==3,1], pen=None, symbol='o', symbolPen=None, symbolSize=7, symbolBrush=(100, 250, 90, 70))
p5a.plot(x= spect_pca[clusterTable==4,0], y = spect_pca[clusterTable==4,1], pen=None, symbol='o', symbolPen=None, symbolSize=7, symbolBrush=(200, 200, 10, 70))


if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
