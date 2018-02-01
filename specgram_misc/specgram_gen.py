'''
thanks to original author delivering core concepts of doing this 

http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html 
'''
import numpy
import scipy.io.wavfile
from scipy.fftpack import dct

#set specgram stats (units in seconds)
frame_size=4.0
frame_stride=0.5


#pre_emphasis: is a artifact from an old system but known to emphasize abit of high freq signals
def preemp_sig(signal, preemp=0.97): #preemp = typically 0.95 or 0.97
    emphasized_signal = numpy.append(signal[0], signal[1:] - preemp * signal[:-1])
    return emphasized_signal

#open .wav file
sample_rate, signal = scipy.io.wavfile.read('f013.wav')  # File assumed to be in the same directory
emphasized_signal=preemp_sig(signal)

test10sec=emphasized_signal[:sample_rate*5]

#framing
frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
signal_length = len(test10sec)
frame_length = int(round(frame_length))
frame_step = int(round(frame_step))
num_frames = int(numpy.ceil(float(numpy.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame



'''adding a pad for the last part of the signal: this part is not used since we do not use last part of the signal


pad_signal_length = num_frames * frame_step + frame_length
z = numpy.zeros((pad_signal_length - signal_length))
pad_signal = numpy.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

'''

indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
frames = test10sec[indices.astype(numpy.int32, copy=False)]
print("\temp sig shape")
print(test10sec.shape)
print("\tindices shape")
print("\t", indices.shape)

#windowing: hamming window is popular
NFFT=1024
frames *= numpy.hamming(frame_length) # frames *= 0.54 - 0.46 * numpy.cos((2 * numpy.pi * n) / (frame_length - 1))  # Explicit Implementation **
mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))  # Magnitude of the FFT
pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

#Filter Bank
nfilt = 1024
low_freq_mel = 0
high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)

fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))
for m in range(1, nfilt + 1):
    f_m_minus = int(bin[m - 1])   # left
    f_m = int(bin[m])             # center
    f_m_plus = int(bin[m + 1])    # right

    for k in range(f_m_minus, f_m):
        fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
    for k in range(f_m, f_m_plus):
        fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
filter_banks = numpy.dot(pow_frames, fbank.T)
filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
filter_banks = 20 * numpy.log10(filter_banks)  # dB

# Mean normalization
filter_banks -= (numpy.mean(filter_banks, axis=0) + 1e-8)
print("\tfilterbank shape")
print(filter_banks.shape)

specgram=filter_bank.T @ pow_frames 


def write_specgram_img(specgram, imgname):   #jpgname with .png
#specgram here has the shape = (1024,1024,2)
    fig, ax = plt.subplots(nrows=1,ncols=1)
    #plt.axis("scaled")
    cax = ax.matshow(np.transpose(specgram), interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot, origin='lower')
    fig.colorbar(cax)
    plt.title('upper: generated_ensemble\n middle:original_ensemble\nlower: vocal_only')
    plt.savefig(imgname, dpi="figure", bbox_inches="tight")
