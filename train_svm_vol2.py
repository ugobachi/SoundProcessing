import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics
import librosa

import make_dataframe


def get_labelname():
    """[summary]
    training下のラベル名を取得
    Returns:
        [type]: [description]
    """    
    current = os.getcwd()
    filepath = current + '/training/'
    print(filepath)
    labellist = []
    for dir in os.listdir(filepath):
        if os.path.isdir(os.path.join(filepath, dir))==True:
            labellist.append(dir)
    
    return labellist, filepath

def wav2list(p):
    """[summary]
    Get audio file list to process all at once
    Returns:
        list : list of audio path
    """
    p = Path(p)
    audio_list = list(p.glob('*.wav'))

    if len(audio_list) == 0:
        sys.exit('Not found in {}'.format(p))

    return audio_list

def get_mfcc_librosa(p):
    """[summary]
    librosaライブラリを用いて24次元MFCCを抽出する
    データはtraining以下に置き, 各ラベルごとにフォルダを作ってデータを置いておく
    Args:
        p ([str]): .wavデータが置いてあるディレクトリ名
    Returns:
        [tupple]: (ファイル名, 24次元のMFCC)
    """    
    wavlist = wav2list(p)
    _name = []
    _mfcc = []
    wavlist.sort()

    for wavfile in wavlist:
        y, sr = librosa.core.load(wavfile,sr=None)
        tmp = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=24)
        ceps = tmp.mean(axis=1)
        # print(ceps)
        _name.append(wavfile.stem)
        _mfcc.append(ceps)

    return _name, _mfcc

def make_df():
    """[summary]
    trainディレクトリ下から.wavデータを取ってきて抽出したMFCCとラベルから構成されるデータフレームを作成
    Returns:
        df_new[dataframe]: 学習データのデータフレーム
    """    
    labellist, filepath = get_labelname()
    cols = [x for x in range(24)]
    print(cols)
    cols.append('label')
    print(cols)
    df_new = pd.DataFrame(index = [], columns=cols)
    # print(df_new)
    for label in labellist:
        # print(filepath, label)
        labelpath = filepath + label
        filename, tmp = get_mfcc_librosa(labelpath)
        df = pd.DataFrame(tmp, index=filename)
        df = df.assign(label=label)
        df_new = pd.concat([df_new, df], axis=0)
    
    return df_new

def train():
    """[summary]
    抽出したMFCCとデータのラベルを元にSVMの学習を行う
    """    
    df = make_df()
    # x : 24次元のMFCC特徴量, y : データのラベル
    x = df.iloc[:, 0:24]
    print(x)
    y = df.iloc[:, 24]

    # ラベルを数字に変換
    label = set(y)
    # print(y)
    label_list = list(label)
    label_list.sort()

    for i in range(len(label_list)):
        y[y == label_list[i]] =i

    y = np.array(y, dtype = "int")
    # print(y)

    #教師データとテストデータの境目を決める
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 1)

    # データ標準化して学習
    sc = StandardScaler()
    sc.fit(x_train)
    x_train_std = sc.transform(x_train)
    x_test_std = sc.transform(x_test)

    # SVMインスタンス生成, カーネルはRBF一択っぽい
    model_rbf = SVC(kernel = "rbf", random_state =1)
    model_rbf.fit(x_train_std, y_train)

    pred_rbf_train = model_rbf.predict(x_train_std)
    accuracy_rbf_train =accuracy_score(y_train, pred_rbf_train)
    print("train_result")
    print("RBF : "+ str(accuracy_rbf_train))

    pred_rbf_test = model_rbf.predict(x_test_std)
    accuracy_rbf_test = accuracy_score(y_test, pred_rbf_test)
    print("-"*25)
    print("test_result")
    print("RBF : "+ str(accuracy_rbf_test))

    # predicted = pd.Series(pred_rbf_test)
    

if __name__ == "__main__":
    train()