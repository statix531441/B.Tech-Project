import numpy as np
from scipy.io import wavfile

# User input
duration=5.0
toneFrequency_left=1000 #Hz (20,000 Hz max value)
toneFrequency_right=15000 #Hz (20,000 Hz max value)

# Constants
samplingFrequency=48000

# Generate Tones
time_x=np.arange(0, duration, 1.0/float(samplingFrequency))
toneLeft_y=5*np.cos(2.0 * np.pi * toneFrequency_left * time_x)
toneRight_y=np.cos(2.0 * np.pi * toneFrequency_right * time_x)

# A 2D array where the left and right tones are contained in their respective rows
tone_y_stereo=np.vstack((toneLeft_y, toneRight_y))

# Reshape 2D array so that the left and right tones are contained in their respective columns
tone_y_stereo=tone_y_stereo.transpose()

# Produce an audio file that contains stereo sound
wavfile.write('stereoAudio.wav', samplingFrequency, tone_y_stereo)
print("Done")