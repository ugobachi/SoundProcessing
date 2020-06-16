from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics
from sklearn.utils import shuffle

import train_svm_vol2 as tr_svm

def train():
    """[summary]
    抽出したMFCCとデータのラベルを元にSVMの学習を行う
    """    
    df = tr_svm.make_df()
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

    print(x, y)

    #教師データとテストデータの境目を決める
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 1)

    # データ標準化して学習
    sc = StandardScaler()
    sc.fit(x_train)
    x_train = sc.fit_transform(x_train)
    x_test = sc.fit_transform(x_test)

    # 正則化項:1/C/2*W^2 →C = ∞で正則化0
    clf_sk = LogisticRegression(C=1, max_iter=500, solver="sag", multi_class="auto", tol=1e-10)
    clf_sk.fit(x_train, y_train)

    # 正答率
    acc_train = accuracy_score(y_train, clf_sk.predict(x_train))
    acc_test = accuracy_score(y_test, clf_sk.predict(x_test))

    print("train_result")
    print("Logistic Regression : " + str(acc_train))
    print("-"*25)
    print("test_result")
    print("Logistic Regression : " + str(acc_test))
    # print("acc_train: "+ str(acc_train) + "   acc_test: "+ str(acc_test))


if __name__ == "__main__":
    train()