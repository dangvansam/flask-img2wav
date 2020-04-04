import librosa
from wav2mel_mel2wav.strechableNumpyArray import StrechableNumpyArray
import numpy as np
import os
import scipy.io
import matplotlib.pyplot as plt
from wav2mel_mel2wav.ourLTFATStft import LTFATStft
import ltfatpy
from wav2mel_mel2wav.modGabPhaseGrad import modgabphasegrad
ltfatpy.gabphasegrad = modgabphasegrad
#tái tạo phase từ spectrogram
from wav2mel_mel2wav.numba_pghi import pghi

fft_hop_size = 64//2
fft_window_length = 512
L = 16384//2
clipBelow = -10

def wav2spec(signal):
    anStftWrapper = LTFATStft()
    spectrograms = np.zeros([int(fft_window_length//2+1), int(L/fft_hop_size)], dtype=np.float64)
    #tgrads = np.zeros([int(fft_window_length//2+1), int(L/fft_hop_size)], dtype=np.float64)
    #fgrads = np.zeros([int(fft_window_length//2+1), int(L/fft_hop_size)], dtype=np.float64)

    gs = {'name': 'gauss', 'M': 512}

    realDGT = anStftWrapper.oneSidedStft(signal=signal, windowLength=fft_window_length, hopSize=fft_hop_size)
    spectrogram = anStftWrapper.logMagFromRealDGT(realDGT, clipBelow=np.e**clipBelow, normalize=True)
    # spectrograms[index] = spectrogram  
    # tgradreal, fgradreal = ltfatpy.gabphasegrad('phase', np.angle(realDGT), fft_hop_size, fft_window_length)
    # tgrads[index] = tgradreal /64
    # fgrads[index] = fgradreal /256

    spectrogram = spectrogram/5+1
    spectrogram = spectrogram[:256, :]
    # print('min spectrogram=',np.min(spectrogram))
    # print('max spectrogram=',np.max(spectrogram))
    # print('mean spectrogram=',np.mean(spectrogram))
    #print(spectrogram.shape)
    #exit()
    return spectrogram

def spec2wav(spectrogram):
    generated_signals = np.exp(5*(spectrogram-1)) # Undo preprocessing
    #print(np.zeros([256,1],dtype=np.float64).shape)
    generated_signals = np.concatenate([generated_signals, np.zeros([1,256],dtype=np.float64)], axis=0) #Fill last column of freqs with zeros
    #print(generated_signals.shape)
    # print('min spectrogram=',np.min(generated_signals))
    # print('max spectrogram=',np.max(generated_signals))
    # print('mean spectrogram=',np.mean(generated_signals))

    anStftWrapper = LTFATStft()

    # Compute Tgrad and Fgrad from the generated spectrograms
    tgrads = np.zeros_like(generated_signals)
    fgrads = np.zeros_like(generated_signals)
    gs = {'name': 'gauss', 'M': fft_window_length}
    magSpectrogram = generated_signals
    tgrads, fgrads = ltfatpy.gabphasegrad('abs', magSpectrogram, gs, fft_hop_size)

    logMagSpectrogram = np.log(magSpectrogram.astype(np.float64))
    phase = pghi(logMagSpectrogram, tgrads, fgrads, fft_hop_size, fft_window_length, L, tol=10)
    reconstructed_audio = anStftWrapper.reconstructSignalFromLoggedSpectogram(logMagSpectrogram, phase, windowLength=fft_window_length, hopSize=fft_hop_size)

    #librosa.output.write_wav('a.wav',reconstructed_audios[0],16000)
    #print("reconstructed audios!")
    return reconstructed_audio