#####REFERENCES#####

"""
functions used for spectrogram inversion to audio, namely,  
    def xcorr_offset(x1, x2):
    def invert_spectrogram(X_s, step, calculate_offset=True, set_zero_phase=True):
    def iterate_invert_spectrogram(X_s, fftsize, step, n_iter=10, verbose=True):
are refered from as mentioned below

Under MSR-LA License
Based on MATLAB implementation from Spectrogram Inversion Toolbox
References
----------
D. Griffin and J. Lim. Signal estimation from modified
short-time Fourier transform. IEEE Trans. Acoust. Speech
Signal Process., 32(2):236-243, 1984.
Malcolm Slaney, Daniel Naar and Richard F. Lyon. Auditory
Model Inversion for Sound Separation. Proc. IEEE-ICASSP,
Adelaide, 1994, II.77-80.
Xinglei Zhu, G. Beauregard, L. Wyse. Real-Time Signal
Estimation from Modified Short-Time Fourier Transform
Magnitude Spectra. IEEE Transactions on Audio Speech and
Language Processing, 08/2007.
"""


"""
other functions that ft to make spectrograms from audio is refered from

Python-Spectrograms-MFCC-and-Inversion-checkpoint-checkpoint.ipynb

by 
"""




# Packages we're using
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.io import wavfile
from scipy.signal import butter, lfilter
import scipy.ndimage
import imageio as img
import cv2
#here stands for converting wavfile objs by scipy.io.wavefile into spectrograms

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def overlap(X, window_size, window_step):
    """
    Create an overlapped version of X
    Parameters
    ----------
    X : ndarray, shape=(n_samples,)
        Input signal to window and overlap
    window_size : int
        Size of windows to take
    window_step : int
        Step size between windows
    Returns
    -------
    X_strided : shape=(n_windows, window_size)
        2D array of overlapped X
    """
    if window_size % 2 != 0:
        raise ValueError("Window size must be even!")
    # Make sure there are an even number of windows before stridetricks
    append = np.zeros((window_size - len(X) % window_size))
    X = np.hstack((X, append))

    ws = window_size
    ss = window_step
    a = X

    valid = len(a) - ws
    nw = int((valid) // ss)
    out = np.ndarray((nw,ws),dtype = a.dtype)

    for i in range(nw):
        # "slide" the window along the samples
        start = i * ss
        stop = start + ws
        out[i] = a[start : stop]

    return out


def stft(X, fftsize=128, step=65, mean_normalize=True, real=False,
         compute_onesided=True):
    """
    Compute STFT for 1D real valued input X
    """
    if real:
        local_fft = np.fft.rfft
        cut = -1
    else:
        local_fft = np.fft.fft
        cut = None
    if compute_onesided:
        cut = fftsize // 2
    if mean_normalize:
        X -= X.mean()

    X = overlap(X, fftsize, step)
    
    size = fftsize
    win = 0.54 - .46 * np.cos(2 * np.pi * np.arange(size) / (size - 1))
    X = X * win[None]
    X = local_fft(X)[:, :cut]
    return X


def pretty_spectrogram(d,log = True, thresh= 5, fft_size = 512, step_size = 64):
    """
    creates a spectrogram
    log: take the log of the spectrgram
    thresh: threshold minimum power for log spectrogram
    """
    specgram = np.abs(stft(d, fftsize=fft_size, step=step_size, real=False,
        compute_onesided=True))
  
    if log == True:
        specgram /= specgram.max() # volume normalize to max 1
        specgram = np.log10(specgram) # take log
        specgram[specgram < -thresh] = -thresh # set anything less than the threshold as the threshold
    else:
        specgram[specgram < thresh] = thresh # set anything less than the threshold as the threshold
    
    return specgram

#from here, matlab ported codes of inverting spectrograms

def xcorr_offset(x1, x2):
    x1 = x1 - x1.mean()
    x2 = x2 - x2.mean()
    frame_size = len(x2)
    half = frame_size // 2
    corrs = np.convolve(x1.astype('float64'), x2[::-1].astype('float64'))
    corrs[:half] = -1E30
    corrs[-half:] = -1E30
    offset = corrs.argmax() - len(x1)
    return offset

def invert_spectrogram(X_s, step, calculate_offset=True, set_zero_phase=True):
    size = int(X_s.shape[1] // 2)
    wave = np.zeros((X_s.shape[0] * step + size))
    # Getting overflow warnings with 32 bit...
    wave = wave.astype('float64')
    total_windowing_sum = np.zeros((X_s.shape[0] * step + size))
    win = 0.54 - .46 * np.cos(2 * np.pi * np.arange(size) / (size - 1))

    est_start = int(size // 2) - 1
    est_end = est_start + size
    for i in range(X_s.shape[0]):
        wave_start = int(step * i)
        wave_end = wave_start + size
        if set_zero_phase:
            spectral_slice = X_s[i].real + 0j
        else:
            # already complex
            spectral_slice = X_s[i]

        # Don't need fftshift due to different impl.
        wave_est = np.real(np.fft.ifft(spectral_slice))[::-1]
        if calculate_offset and i > 0:
            offset_size = size - step
            if offset_size <= 0:
                print("WARNING: Large step size >50\% detected! "
                      "This code works best with high overlap - try "
                      "with 75% or greater")
                offset_size = step
            offset = xcorr_offset(wave[wave_start:wave_start + offset_size],
                                  wave_est[est_start:est_start + offset_size])
        else:
            offset = 0
        wave[wave_start:wave_end] += win * wave_est[
            est_start - offset:est_end - offset]
        total_windowing_sum[wave_start:wave_end] += win
    wave = np.real(wave) / (total_windowing_sum + 1E-6)
    return wave

def iterate_invert_spectrogram(X_s, fftsize, step, n_iter=10, verbose=True):
    reg = np.max(X_s) / 1E8
    X_best = copy.deepcopy(X_s)
    for i in range(n_iter):
        if verbose:
            print(("Runnning iter %i" % i))
        if i == 0:
            X_t = invert_spectrogram(X_best, step, calculate_offset=True,
                                     set_zero_phase=True)
        else:
            # Calculate offset was False in the MATLAB version
            # but in mine it massively improves the result
            # Possible bug in my impl?
            X_t = invert_spectrogram(X_best, step, calculate_offset=True,
                                     set_zero_phase=False)
        est = stft(X_t, fftsize=fftsize, step=step, compute_onesided=False)
        phase = est / np.maximum(reg, np.abs(est))
        X_best = X_s * phase[:len(X_s)]
    X_t = invert_spectrogram(X_best, step, calculate_offset=True,
                             set_zero_phase=False)
    return np.real(X_t)

def invert_pretty_spectrogram(X_s, log = True, fft_size = 512, step_size = 512/4, n_iter = 10):
    
    if log == True:
        X_s = np.power(10, X_s)

    X_s = np.concatenate([X_s, X_s[:, ::-1]], axis=1)
    X_t = iterate_invert_spectrogram(X_s, fft_size, step_size, n_iter=n_iter)
    return X_t


### Parameters ###
fft_size = 2048 # window size for the FFT (resolution for freq bin)
step_size = int(fft_size/16) # distance to slide along the window (in time)
spec_thresh = 4 # threshold for spectrograms (lower filters out more noise)
lowcut = 50 # Hz # Low cut for our butter bandpass filter
highcut = 17000 # Hz # High cut for our butter bandpass filter



# Grab your wav and filter it
startpt=int(input("what pt do you want your music start: "))
how_long=int(input("length of the piece: "))
mywav = 'mono256.wav'
rate, data = wavfile.read(mywav)
data = butter_bandpass_filter(data, lowcut, highcut, rate, order=1)
# Only use a short clip for our demo
if np.shape(data)[0]/float(rate) > how_long:
    data = data[startpt*rate:(startpt+how_long)*rate] 
    print(data)
    print(len(data))
print('Length in time (s): ', np.shape(data)[0]/float(rate))

#this is for 10s. if we change 10 into 2 would it be 2s clip as well? yes



#making spectrogram
wav_spectrogram = pretty_spectrogram(data.astype('float64'), fft_size = fft_size, 
                                   step_size = step_size, log = True, thresh = spec_thresh)


#plotting it
fig, ax = plt.subplots(nrows=1,ncols=1)
#fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(20,4))
cax = ax.matshow(np.transpose(wav_spectrogram), interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot, origin='lower')
#fig.colorbar(cax) # if we dispose this line, then the figure would appear w/o colorbar only?
#plt.title('Original Spectrogram')
#plt.show()
plt.savefig("C:/Users/SEONIL/Documents/for_analysis.png",bbox_inches="tight",pad_inches=0)


#spectrogram in raw view?
#imported imageio as img
cv2.imwrite("C:/Users/SEONIL/Documents/cv2.jpg",wav_spectrogram)   #doing this just converts whole contents into 0. normalize the contents and save again!
a= cv2.imread("C:/Users/SEONIL/Documents/cv2.jpg",0)
print("row of pic (time)", len(a))
print("col of pic (#f-bin)", len(a[0]))#f**k yeaH I cracked this from bottom! so much primitive way I guess

'''
img.imwrite("C:/Users/SEONIL/Documents/imgio.jpg",np.transpose(wav_spectrogram))
a=cv2.imread("C:/Users/SEONIL/Documents/test_cv2.jpg", 0)
f0=a.copy()
f0=cv2.flip(a, 0)
cv2.imwrite("C:/Users/SEONIL/Documents/nowdone.png",f0)
'''

with open("C:/Users/SEONIL/Documents/logger.txt", "w") as logger:
    np.set_printoptions(threshold=np.nan)
#    logger.write(str(a))
#np.savetxt("C:/Users/SEONIL/Documents/logger.csv", wav_spectrogram)



# Invert from the spectrogram back to a waveform
recovered_audio_orig = invert_pretty_spectrogram(wav_spectrogram, fft_size = fft_size,
                                            step_size = step_size, log = True, n_iter = 10)



'''audio wav files are just amp vs time 1D array (with samplying rate. 44100 elements corresponds to 1s)
print(type(recovered_audio_orig))
print(recovered_audio_orig.shape)
print(recovered_audio_orig)
print(len(recovered_audio_orig))
'''



#normalize
recovered_audio_orig/=max(recovered_audio_orig)
recovered_audio_orig*=3
#truncate -- because of inconsistency of the recovered array length with original one. 
#encoding never had been a problem. just groove player sucks at reading those


wavfile.write('C:/Users/SEONIL/Documents/not_trunc.wav', 44100, recovered_audio_orig)
#wavfile.write('C:/Users/SEONIL/Documents/trunc_tail.wav', 44100, recovered_audio_orig1)
#wavfile.write('C:/Users/SEONIL/Documents/trunc_head.wav', 44100, recovered_audio_orig2)
#IPython.display.Audio(data=recovered_audio_orig, rate=rate) # play the audio