import numpy as np
import pandas as pd
import mfcc


def get_mfcc(p):
    """[summary]
    Args:
        p ([str]): .wavデータが置いてあるディレクトリ名
    Returns:
        [tupple]: (ファイル名, 13次元のMFCC)
    """    
    wavlist = mfcc.wav2list(p)
    _name = []
    _mfcc = []
    wavlist.sort()

    for wavfile in wavlist:
        tmp = mfcc.get_feature(str(wavfile), nfft=2048, nceps=13)
        _name.append(wavfile.stem)
        _mfcc.append(tmp)

    return _name, _mfcc

def make_df():
    """[summary]
    データフレーム作成
    Returns:
        df[dataframe]: 取得したファイル名とMFCCのリストからデータフレーム作成
    """    
    p = input('Please input sound type : ')
    filename, tmp = get_mfcc(p)
    df = pd.DataFrame(tmp, index=filename)
    return df


if __name__ == "__main__":
    print(make_df())