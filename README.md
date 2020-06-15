# SoundProcessing

## 概要
音声周りの処理を色々まとめる<br>
・MFCCの取得(mfcc.py, plotting.py)<br>
・soxコマンドからスペクトログラムを取得(shell_spec.py)<br>
・取得したMFCCからpandasでデータフレームを作成する(make_dataframe.py)<br>
・SVMの学習と認識その1(train_svm_vol1.py)<br>
・SVMの学習と認識その2(train_svm_vol2.py)<br>
※SVMの学習は自前の学習データを用意する必要あり<br>

## 実行環境
### pythonのバージョン
python3.6
### 使用したライブラリ
numpy1.18.1 <br>
scipy1.4.1  <br>
matplotlib3.1.3 <br>
pandas1.0.3<br>


## 参考
・ Webデータリポート - 生活音を機械学習してみた ( http://webdatareport.hatenablog.com/entry/2016/11/06/161304 )<br>
・ 音声波形のサンプルデータ - ( https://wsignal.sakura.ne.jp/onsei2007/wav_data51/wav_data51.html )<br>
・ soxコマンドの使い方等 - ( http://sox.sourceforge.net/sox.html )<br>
