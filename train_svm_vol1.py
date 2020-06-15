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

import mfcc
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

def get_mfcc(p):
    """[summary]
    Args:
        p ([str]): .wavデータが置いてあるディレクトリ名
    Returns:
        [tupple]: (ファイル名, 13次元のMFCC)
    """    
    wavlist = wav2list(p)
    _name = []
    _mfcc = []
    wavlist.sort()

    for wavfile in wavlist:
        tmp = mfcc.get_feature(str(wavfile), nfft=8192, nceps=13)
        _name.append(wavfile.stem)
        _mfcc.append(tmp)

    return _name, _mfcc

def make_df():
    """[summary]
    trainディレクトリ下から.wavデータを取ってきて抽出したMFCCとラベルから構成されるデータフレームを作成
    Returns:
        df_new[dataframe]: 学習データのデータフレーム
    """    
    labellist, filepath = get_labelname()
    cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 'label']
    df_new = pd.DataFrame(index = [], columns=cols)
    # print(df_new)
    for label in labellist:
        # print(filepath, label)
        labelpath = filepath + label
        filename, tmp = get_mfcc(labelpath)
        df = pd.DataFrame(tmp, index=filename)
        df = df.assign(label=label)
        df_new = pd.concat([df_new, df], axis=0)
    
    return df_new

def train():    
    df = make_df()
    # x : 1~13次元のMFCC特徴量, y : データのラベル
    x = df.iloc[:, 0:13]
    print(x)
    y = df.iloc[:, 13]

    # ラベルを数字に変換
    label = set(y)
    print(y)
    label_list = list(label)
    label_list.sort()

    for i in range(len(label_list)):
        y[y == label_list[i]] =i

    y = np.array(y, dtype = "int")
    print(y)

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
    print("-"*30)
    print("test_result")
    print("RBF : "+ str(accuracy_rbf_test))

    predicted = pd.Series(pred_rbf_test)
    print(predicted)
    

if __name__ == "__main__":
    train()