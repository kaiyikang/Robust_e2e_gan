import os
import sys

from numpy.core.fromnumeric import argmax
import kaldi_io
import librosa
import numpy as np
import scipy.signal
import torchaudio
import math
import random
from random import choice
from multiprocessing import Pool
import scipy.io.wavfile as wav
import pdb

root_path=os.path.join(os.path.abspath('.'))   # 表示当前所处的文件夹的绝对路径
# 负责生成noise信号的函数
# 由于作者没有在readme中，没写明白clean_feat.scp的组成
# 因此可以在这个文件中，反推输入scp是什么

'''def load_audio(path):
    sound, _ = torchaudio.load(path, normalization=True)
    sound = sound.numpy()
    if len(sound.shape) > 1:
        if sound.shape[1] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis=1)  # multiple channels, average
    sound = sound / 65536.
    return sound'''

def load_audio(path):
    path = os.path.join(root_path,path)
    print(path)
    rate, sig = wav.read(path)
    sig = sig.astype(np.float32)
    return sig
    
    
def MakeMixture(speech, noise, db):
    if speech is None or noise is None:
        return None
    if np.sum(np.square(noise)) < 1.0e-6:
        return None

    spelen = speech.shape[0]

    exnoise = noise
    while exnoise.shape[0] < spelen:
        exnoise = np.concatenate([exnoise, noise], 0)
    noise = exnoise
    noilen = noise.shape[0]
    
    elen = noilen - spelen - 1
    if elen > 1:
        s = round(random.randint(0, elen - 1))
    else:
        s = 0
    e = s + spelen    
    
    noise = noise[s:e]
    
    try:
        noi_pow = np.sum(np.square(noise))
        if noi_pow > 0:
            noi_scale = math.sqrt(np.sum(np.square(speech)) / (noi_pow * (10 ** (db / 10.0))))
        else:
            print(noi_pow, np.square(noise), 'error')
            return None
    except:
        return None

    nnoise  = noise * noi_scale
    mixture = speech + nnoise
    mixture = mixture.astype('float32')
    return mixture


def make_feature(wav_path_list, noise_wav_list, feat_dir, thread_num, argument=False, repeat_num=1):
    mag_ark_scp_output = 'ark:| copy-feats --compress=true ark:- ark,scp:{0}/feats{1}.ark,{0}/feats{1}.scp'.format(feat_dir, thread_num)
    ang_ark_scp_output = 'ark:| copy-feats --compress=true ark:- ark,scp:{0}/angles{1}.ark,{0}/angles{1}.scp'.format(feat_dir, thread_num)
    
    if argument:
        fwrite = open(os.path.join(feat_dir, 'db' + str(thread_num)), 'a')


    # f_mag = kaldi_io.open_or_fd(mag_ark_scp_output,'wb')
    # f_ang = kaldi_io.open_or_fd(ang_ark_scp_output,'wb')
    print("进入num循环")
    for num in range(repeat_num):
        for tmp in wav_path_list:
            uttid, wav_path = tmp
            clean = load_audio(wav_path)
            y = None
            while y is None:
                if argument:
                    print("argument=True")
                    noise_path = choice(noise_wav_list)
                    n = load_audio(noise_path[1])
                    db = np.random.uniform(low=0, high=20)
                    y = MakeMixture(clean, n, db)
                    uttid_new = uttid + '__mix{}'.format(num)
                    print(uttid_new + ' ' + str(db) + '\n')
                    fwrite.write(uttid_new + ' ' + str(db) + '\n')
                else:
                    
                    y = clean
                    uttid_new = uttid
            # STFT
            print("y = ", y)
            if y is not None:
                D = librosa.stft(y, n_fft=512, hop_length=256, win_length=512, window=scipy.signal.hamming)
                spect = np.abs(D)
                angle = np.angle(D)
                ##feat = np.concatenate((spect, angle), axis=1)
                ##feat = feat.transpose((1, 0))
                with kaldi_io.open_or_fd(mag_ark_scp_output,'wb') as f:
                    kaldi_io.write_mat(f, spect.transpose((1, 0)), key=uttid_new)

                with kaldi_io.open_or_fd(ang_ark_scp_output,'wb') as f:
                    kaldi_io.write_mat(f, angle.transpose((1, 0)), key=uttid_new)
            else:
                print(noise_path, tmp, 'error')
    
    if argument:
        fwrite.close()

        
def main():
    
    # 输入参数
    # data_dir = sys.argv[1]
    # feat_dir = sys.argv[2]
    # noise_repeat_num = int(sys.argv[3])
    data_dir = "/home/kang/Develop/Robust_e2e_gan/data/data_aishell" 
    feat_dir = "/home/kang/Develop/Robust_e2e_gan/data/small_feats" 
    noise_repeat_num = 1
    
    # 建立clean数据的路径
    clean_feat_dir = os.path.join(feat_dir, 'clean')
    if not os.path.exists(clean_feat_dir):
        os.makedirs(clean_feat_dir)

    # 建立mix数据的路径
    mix_feat_dir = os.path.join(feat_dir, 'mix')
    if not os.path.exists(mix_feat_dir):
        os.makedirs(mix_feat_dir)
    
    clean_wav_list = []
    clean_wav_scp = os.path.join(data_dir, 'clean_wav.scp') # 在data_dir 目录下，有一个clean_wav.scp 文件
    with open(clean_wav_scp, 'r', encoding='utf-8') as fid:
        for line in fid:
            line = line.strip().replace('\n','')
            uttid, wav_path = line.split()
            clean_wav_list.append((uttid, wav_path)) 
    print('clean_wav_list', len(clean_wav_list)) 
    
    noise_wav_list = []
    noise_wav_scp = os.path.join(data_dir, 'noise.scp')
    with open(noise_wav_scp, 'r', encoding='utf-8') as fid:
        for line in fid:
            line = line.strip().replace('\n','')
            wav_path = line.split()
            noise_wav_list.append(wav_path) 
    print('noise_wav_list', len(noise_wav_list)) 
    
    # 使用八个线程
    threads_num = 2
    wav_num = len(clean_wav_list)
    print('Parent process %s.' % os.getpid())
    p = Pool()    
    start = 0
    for i in range(threads_num):
        end = start + int(wav_num / threads_num)
        if i == (threads_num - 1):
            end = wav_num
        wav_path_tmp_list = clean_wav_list[start:end]
        start = end
        p.apply_async(make_feature, args=(wav_path_tmp_list, noise_wav_list, clean_feat_dir, i, False))
        print('apply')
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')    
    command_line = 'cat {}/feats*.scp > {}/clean_feats.scp'.format(clean_feat_dir, data_dir)   
    os.system(command_line) 
    command_line = 'cat {}/angles*.scp > {}/clean_angles.scp'.format(clean_feat_dir, data_dir)   
    os.system(command_line) 
    
    print('\n')

    wav_num = len(clean_wav_list)
    print('Parent process %s.' % os.getpid())
    p = Pool()    
    for i in range(threads_num):
        wav_path_tmp_list = clean_wav_list[int(i * wav_num / threads_num): int((i + 1) * wav_num / threads_num)]
        p.apply_async(make_feature, args=(wav_path_tmp_list, noise_wav_list, mix_feat_dir, i, True, noise_repeat_num))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')  
      
    command_line = 'cat {}/feats*.scp > {}/mix_feats.scp'.format(mix_feat_dir, data_dir) 
    print(command_line)  
    os.system(command_line) 
    command_line = 'cat {}/angles*.scp > {}/mix_angles.scp'.format(mix_feat_dir, data_dir)   
    os.system(command_line) 
    print(command_line)  

    command_line = 'cat {}/db* > {}/db.scp'.format(mix_feat_dir, data_dir)   
    os.system(command_line)        
    print(command_line)  
  
if __name__ == '__main__':
    main()