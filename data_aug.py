import sys
import os
from pathlib import Path
import librosa

import train_svm_vol2

def wav2list(p):
    """[summary]
    Get audio file list to process all at once
    Returns:
        list : list of audio path
    """
    p = Path(p)
    audio_list = list(p.rglob('*.wav'))

    if len(audio_list) == 0:
        sys.exit('Not found in {}'.format(p))

    return audio_list

def parentpath(path=__file__, f=0):
    """[summary]
    取得した音声ファイルのディレクトリの絶対パスを取得
    Args:
        path (str, optional): 音声ファイルのパス. Defaults to __file__.
        f (int, optional): 0の時は音声ファイルの親ディレクトリまでの絶対パス. Defaults to 0.

    Returns:
        [str]: 親ディレクトリの絶対パス
    """    
    return str('/'.join(os.path.abspath(path).split('/')[0:-1-f]))

def stretch(data, rate):
    """[summary]
    time_stretchで速度調整, 時間を短くするときはスピードが早くなる
    Args:
        data ([type]): 調整前のオリジナル音声データ
        rate (int, optional): 調整の割合. Defaults to 1.

    Returns:
        [type]: 音声データの長さをrate倍した音声データ
    """
    data = librosa.effects.time_stretch(data, rate)
    return data

def aug():
    """[summary]
    ディレクトリ名と速度調整のレートを入力してデータ拡張を行う
    """    
    p = input('Please input directory name : ')
    rate = float(input('Please input speed rate : '))
    audio_list = wav2list(p)

    for i in audio_list:
        if 'aug' in i.stem:
            print('{} is skipped'.format(i.stem))
            pass
        else:
            data, sr = librosa.core.load(i, sr=None)
            aug_data = stretch(data, rate)
            name = i.stem + '_speedaug_' + str(int(rate*100)) + 'per.wav'
            output_path = parentpath(str(i))
            output_path = output_path + '/' + name
            librosa.output.write_wav(output_path, aug_data, sr=sr)


if __name__ == "__main__":
    aug()