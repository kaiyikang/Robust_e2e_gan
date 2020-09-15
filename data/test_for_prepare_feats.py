
import kaldi_io
import scipy.io.wavfile as wav
import librosa
import numpy as np
import scipy.signal
import scipy.io.wavfile as wav

def kang():
    for i in range(2):
        f_mag = 'ark:| copy-feats --compress=true ark:- ark,scp:file1.ark,file2.scp'

        audop_path = '/home/kang/Develop/Robust_e2e_gan/db/data_aishell/wav/dev/S0724/BAC009S0724W0121.wav'
        rate, sig = wav.read(audop_path)
        sig = sig.astype('float32')
        D = librosa.stft(sig, n_fft=512, hop_length=256, win_length=512, window=scipy.signal.hamming)
        spect = np.abs(D)
        uttid_new = "file2deID"

        kaldi_io.write_mat(f_mag, spect.transpose((1, 0)), key=uttid_new)

# %%
kang()