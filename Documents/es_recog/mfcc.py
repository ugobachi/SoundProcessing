'''
MFCCの抽出手順
1. プリエンファシスフィルタで波形の高域成分を強調する
2. 窓関数をかけた後にFFT（高速フーリエ変換）して振幅スペクトルを求める
3. 振幅スペクトルにメルフィルタバンクをかけて圧縮する
4. 上記の圧縮した数値列を信号とみなして離散コサイン変換する
5. 得られたケプストラムの低次成分(低次から13次元)をMFCCとする
'''

from pathlib import Path

import sys
import wave
import numpy as np
import scipy.signal
import scipy.fftpack
import scipy.fftpack.realtransforms


def wav2list(p):
    """[summary]
    Get audio file list to process all at once
    Returns:
        list : list of audio path
    """
    p = Path('./' + p)
    audio_list = list(p.glob('*.wav'))

    if len(audio_list) == 0:
        sys.exit('Not found in {}'.format(p))

    return audio_list

def wavread(wav):
    """[summary]
    Opening .wav file and getting necessary information
        
    Args:
        wav (str): target of reading file(.wav)

    Returns:
        tupple (x, float(fs)): (frame count, framerate)
    """
    wf = wave.open(wav, "r")
    fs = wf.getframerate()
    x  = wf.readframes(wf.getnframes())
    x  = np.frombuffer(x, dtype="int16") / 32768.0
    wf.close()
    return x, float(fs)

def hz2mel(f):
    """[summary]
    Conversion frequency[Hz] to mel frequency[Hz]

    Args:
        f (float): frequency

    Returns:
        float : mel frequency
    """    
    return 1127.01048 * np.log(f / 700.0 + 1.0)

def mel2hz(m):
    """[summary]
    Conversion mel frequency[Hz] to frequency[Hz]

    Args:
        m (float): mel frequency

    Returns:
        float: frequency
    """    
    return 700.0 * (np.exp(m / 1127.01048) - 1.0) 


def melfilterbank(fs, nfft, numChannels):
    """[summary]
    Creating mel filterbank

    Args:
        fs (float): frequency[Hz]
        nfft (int): sampling count of FFT
        numChannels (int): channel count of melfilterbank

    Returns:
        tuple (filterbank, fcenters): [description]
    """    
    fmax = fs / 2
        
    melmax = hz2mel(fmax)
        
    nmax = nfft // 2
        
    df = fs / nfft
        
    dmel = melmax / (numChannels + 1)
        
    # np.arange() : To create Arithmetic sequence　or sequence number
    melcenters = np.arange(1, numChannels + 1) * dmel

    fcenters = mel2hz(melcenters)

    # np.round() : Rounding
    indexcenter = np.round(fcenters / df)
    # np.hstack : to combine array
    indexstart = np.hstack(([0], indexcenter[0:numChannels - 1]))
    indexend = np.hstack((indexcenter[1:numChannels], [nmax]))

    filterbank = np.zeros((numChannels, nmax))

    for c in np.arange(0, numChannels):
        # left side of triangle filter
        increment = 1.0 / (indexcenter[c] - indexstart[c])
        for i in np.arange(indexstart[c], indexcenter[c]):
            i = int(i)
            filterbank[c, i] = (i - indexstart[c]) * increment

        # right side of triangle filter
        decrement = 1.0 / (indexend[c] - indexcenter[c])
        for i in np.arange(indexcenter[c], indexend[c]):
            i = int(i)
            filterbank[c, i] = 1.0 - ((i - indexcenter[c]) * decrement)
        
    return filterbank, fcenters


def preEmphasis(signal, p):
    """[summary]
    To create preemphasis filter
    scipy.signal.ifilter(numerator coefficient, denominator coefficient, input signal) : create FIR filter
    Args:
        signal (array): input signal
        p (float): preemphasis coefficient

    Returns:
        list : array of signal after filtering
    """    
    return scipy.signal.lfilter([1.0, -p], 1, signal)


def get_mfcc(signal, nfft, fs, nceps):
    """[summary]

    Args:
        signal (array): input signal
        nfft (int): sampling num of FFT(1024, 2048, 4096, ...)
        fs ([type]): [description]
        nceps (int): dimention count of MFCC

    Returns:
        array : coefficient array from 1 dim to [nceps] dim
    """    
    # filtering by preemphasis filter
    signal = preEmphasis(signal, p=0.97)

    # filtering by hamming window
    hammingWindow = np.hamming(len(signal))
    signal *= hammingWindow

        
    spec = np.abs(np.fft.fft(signal, nfft))[:nfft//2]
    # fscale = np.fft.fftfreq(nfft, d = 1.0 / fs)[:nfft//2]

    # To create melfilterbank
    filterbank, fcenters = melfilterbank(fs, nfft, numChannels=20)

    mspec = np.log10(np.dot(spec, filterbank.T))

    # Diecrete cosine transform
    ceps = scipy.fftpack.realtransforms.dct(mspec, type=2, norm="ortho", axis=-1)

    return ceps[:nceps]


def get_feature(wavfile, nfft, nceps):
    """[summary]

    Args:
        wavfile ([type]): [description]

    Returns:
        [type]: [description]
    """    

    wav, fs = wavread(wavfile)
    # t = np.arange(0.0, len(wav) / fs, 1 / fs)

    '''
    If you want to processing Environment Sound, you can cut out a part of signal array
    center = len(wav) / 2
    cuttime = 0.8
    '''
    # wavdata = wav[int(center - cuttime/2*fs) : int(center + cuttime/2*fs)]

    ceps = get_mfcc(wav, nfft, fs, nceps)

    # tolost() : several num managed by list
    return ceps.tolist()

