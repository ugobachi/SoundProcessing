import os 
import sys
from pathlib import Path

def wav2list_abs(p):
    """[summary]
    Get audio file list to process all at once
    Returns:
        list : list of audio path
    """
    p = Path('./' + p)
    # 絶対パスで取得
    audio_list = list(p.resolve().glob('*.wav'))

    if len(audio_list) == 0:
        sys.exit('Not found in {}'.format(p))

    return audio_list


if __name__ == "__main__":
    p = input('Please input sound name : ')
    wavlist = wav2list_abs(p)
    print(wavlist)
    # 新たにディレクトリを作成してそこにスペクトログラムを入れていく
    dir_name = p + '_spectrogram'
    os.makedirs(dir_name, exist_ok=True)
    # chdirで移動する
    os.chdir(dir_name)
    # os.system('ls')

    for wavdata in wavlist:
        wavname = str(wavdata.stem)
        wavdata = str(wavdata)
        command = 'sox ' + wavdata + ' -n spectrogram -o ' + wavname + '.png'
        # print(command)
        os.system(command)