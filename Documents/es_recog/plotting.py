import mfcc
from matplotlib import pyplot as plt
from matplotlib import cm

def mfcc_plot():
    p = input('Please input sound type : ')
    wavlist = mfcc.wav2list(p)

    for wavfile in wavlist:
        tmp = mfcc.get_feature(str(wavfile), nfft=2048, nceps=13)
        print(wavfile, tmp)
        plt.plot(tmp)
    
    plt.show()

if __name__ == "__main__":
    mfcc_plot()