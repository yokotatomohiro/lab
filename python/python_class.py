# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys
"""
IsingModel={
   # 設定について
   NumInte : 変数の整数部のビット数
   NumDec  : 変数の小数部のビット数
   bits    : 一変数あたりのビット数
   Type    : 問題のタイプ 連続変数：'V', グラフ：'G'

   # 問題について
   X       : 問題の行列 [DIM,NumData]
   y       : 正解のラベル [NumData]
   DIM     : 問題の次元数
   NumData : 問題のサンプル数

   # 設定と問題から作成されるイジングモデル
   J       : 二体相互作用の上三角行列  (DIM×DIM)
   h       : 一体相互作用のベクトル　  (DIM×DIM)
   J_lasso : 二体相互作用の上三角行列  ((DIM*3)×(DIM*3))
   h_lasso : 一体相互作用のベクトル　  ((DIM*3)×(DIM*3))
"""

# データセットおよび詳細な条件についての設定を行う
class Detale():
    def __init__(self): # 各インスタンス変数の初期化
        Detale.NumInte = 7                           # 整数部のビット数
        Detale.NumDec  = 4                           # 小数部のビット数
        Detale.bits    = self.NumInte+self.NumDec+1  # 1変数あたりに使用するビット数
        Detale.Type    = 'V'                         # 問題のタイプを表す 'V':'連続変数', 'G':'グラフ'
        Detale.J = None # 二体相互作用の係数
        Detale.J_lasso = None
        Detale.h = None # 一体相互作用の係数
        Detale.h_lasso = None

    def Change_NumInte(self, Num): # 整数部のビット数を変更
        if (Detale.Type == 'V'):
            Detale.NumInte = Num
            Detale.bits = Detale.NumInte+Detale.NumDec+1
            print('整数部：%dbits, 小数部：%dbits' % (Detale.NumInte, Detale.NumDec))

    def Change_NumDec(self, Num): # 小数部のビット数を変更
        if (selfx.Type == 'V'):
            Detale.NumDec = Num
            Detale.bits = Detale.NumInte+Detale.NumDec+1
            print('整数部：%dbits, 小数部：%dbits' % (Detale.NumInte, Detale.NumDec))

    def Change_Type(self): # 問題のタイプを変更する
        if (Detale.Type == 'V'): # 連続変数 -> グラフ
            Detale.Type = 'G'
            Detale.NumInte = 0
            Detale.NumDec = 0
            Detale.bits = 1
            print('問題のタイプをグラフに変更しました.')
        else:                  # グラフ -> 連続変数
            Detale.Type = 'V'
            Detale.NumInte = 7
            Detale.NumDec = 4
            Detale.bits = Detale.NumInte+Detale.NumDec+1
            print('問題のタイプを連続変数に変更しました.')
    
    # DIM次元のデータセット'X'と正解のラベル'y'を作成する データ数はデフォルトで100個 
    def Set_DataSet(self, DIM=5, NumData=100, Filename=None):
        if Filename is None: # ファイル名に指定がない場合
            Detale.DIM = DIM
            Detale.NumData = NumData
            Detale.X = np.zeros((DIM, NumData))
            Detale.y = None
            answer = np.random.normal(0, scale=32, size=DIM)
            print(answer)
            
            for i in range(DIM):
                Detale.X[i,:] = np.random.normal(0, scale=1, size=NumData)
                if Detale.y is None:
                    Detale.y  = answer[i]*Detale.X[i,:]
                else:
                    Detale.y += answer[i]*Detale.X[i,:]
        else: # ファイル名に指定がある場合 .csvファイルに限る
            df = pd.read_csv(Filename)
            Detale.DIM = df.shape[1]-1
            Detale.NumData = df.shape[0]
            Detale.X = np.zeros((DIM, NumData))
            Detale.y = [NumData]
            for i in range(NumData):
                for j in range(DIM):
                    Detale.X[j][i] = df.iat[j][i]
                Detale.y[i] = df.iat[DIM+1][i]
                
    def Set_CostFunction(self, Lambda=1, func='LS'): # type=lassoとすると追加でJ_lasso, h_lassoを作成する
        if Detale.J is None:
            Detale.J = np.zeros((Detale.DIM, Detale.DIM))
        J_sub = Detale.X**2
        for i in range(Detale.DIM): # Jの対角成分を埋める
            Detale.J[i][i] += np.sum(J_sub[i,:])
        if Detale.h is None:
            Detale.h = np.zeros(Detale.DIM) 
        for i in range(Detale.DIM):
            Detale.h[i] += -2*np.sum(Detale.X[i,:]*Detale.y) # hの成分を埋める
        if func is 'lasso':
            if Detale.J_lasso is None:
                Detale.J_lasso = np.zeros((Detale.DIM*3, Detale.DIM*3)) # 変数はx,m_1,m_2の順
            for i in range(Detale.DIM):
                Detale.J_lasso[i*3][i*3] += Detale.J[i][i]
                # l-1normの部分
                for j in range(3): # 制約項
                    Detale.J_lasso[i*3+j][i*3+j] += Lambda
                Detale.J_lasso[i*3][i*3+1]   +=  2*Lambda # m*z_1
                Detale.J_lasso[i*3][i*3+2]   += -2*Lambda # m*z_2
                Detale.J_lasso[i*3+1][i*3+2] += -2*Lambda # z_1*z_2
            if Detale.h_lasso is None:
                Detale.h_lasso = np.zeros(Detale.DIM*3)
            for i in range(Detale.DIM):
                Detale.h_lasso[i*3]   += Detale.h[i]
                Detale.h_lasso[i*3+1] += Lambda
                Detale.h_lasso[i*3+2] += Lambda

class Qbits(Detale):
    def __init__(Detale):
        super().__init__(self)
                
if __name__ == '__main__':
    import mglearn

    # X[DIM,NumData]:問題のデータセット, y[NumData]:問題のラベル, A[NumData]:問題の解
    model = Detale()
    model.Set_DataSet(DIM=2)
    model.Set_CostFunction(func='lasso')
    print(Detale.J_lasso)
    print(Detale.h_lasso)
    print('グラフ問題を作成する')
    DIM = 5 # 頂点の数を設定する     
