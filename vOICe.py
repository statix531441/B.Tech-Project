import numpy as np
from scipy.io import wavfile
import cv2
import matplotlib.pyplot as plt

samplingFrequency = 48000

def voice(file, tones, res=(512,256)):
    print("Works")
    dmap = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
    dmap = cv2.resize(dmap, res)
    dmap = dmap/dmap.max()

    for i in range(res[0]):
        col = dmap.T[i]
        amps = col.reshape(col.shape[0],1)
        sig = (amps*tones).sum(axis=0)
        # if sig.max()>0:
        #     sig = sig/sig.max()

        t = i/res[0]

        col_stereo = np.vstack(((1-t)*sig, t*sig))

        if i==0:
            output = col_stereo
        else:
            output = np.hstack((output,col_stereo))

    return output.transpose()/output.max()
