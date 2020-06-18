import os
import wave
from pydub import AudioSegment

def makedir(dir_name):
    """[summary]
    出力先のディレクトリを作成し、パスを返す
    Args:
        dir_name (str): 作成したいディレクトリの名前

    Returns:
        [str]: 作成したディレクトリの絶対パス
    """    
    os.makedirs(dir_name, exist_ok=True)
    current = os.getcwd()
    current = current + '/' + dir_name
    return current

def snip_audio():
    """[summary]
    .wavファイルを1秒ごとに区切ってoutputdirフォルダに保存する
    """        
    name = input('Please input file name : ')
    target = './' + name + '.wav'
    outputdir = makedir('sniptest')
    # AudioFileの読み込み
    target_sound = AudioSegment.from_wav(target)
    time = target_sound.duration_seconds * 1000
    src = 0
    dist = 1000
    nt_src = 0
    nt_dist = 1
    while dist < time:
        snip = target_sound[src:dist]
        snip.export(outputdir + '/snip' + name + str(nt_src) + 'sto' + str(nt_dist) + 's' + '.wav', format='wav')
        src = dist
        dist += 1000
        if dist >= time:
            snip_last = target_sound[src:-1]
            snip_last.export(outputdir + '/snip' + name + str(nt_src) + 'stoend' + '.wav', format='wav')
        nt_src += 1
        nt_dist += 1


if __name__ == "__main__":
    snip_audio()