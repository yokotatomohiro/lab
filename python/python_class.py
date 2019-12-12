# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys
"""
Quantum_MonteCarlo={
   # 設定について
   NumInte : 変数の整数部のビット数
   NumDec  : 変数の小数部のビット数
   bits    : 一変数あたりのビット数
   Type    : 問題のタイプ 連続変数：'V', グラフ：'G'

   # 問題について
   X       : 問題の行列
   y       : 正解のラベル
   DIM     : 問題の次元数
   NumData : 問題のサンプル数

   # コスト関数の設定
   Func    : コスト関数について

   # 設定と問題から作成されるイジングモデル
   H       : 二体および一体の相互作用を持つ行列 (DIM×DIM)
   H_B     : 二体および一体の相互作用を持つ行列 ((DIM*bits)×(DIM*bits))
"""

#class Annealing()

class Quantum_MonteCarlo: # 量子モンテカルロアルゴリズムの計算を行う
    def Set_CostFunction(self):
        if self.H is None:
            self.H = np.zeros((self.bits, self.bits))
        
    
    def __init__(self): # 各インスタンス変数の初期化
        self.NumInte = 7                           # 整数部のビット数
        self.NumDec  = 4                           # 小数部のビット数
        self.bits    = self.NumInte+self.NumDec+1  # 1変数あたりに使用するビット数
        self.Type    = 'V'                         # 問題のタイプを表す 'V':'連続変数', 'G':'グラフ'
        self.H       = None

    def Change_NumInte(self, Num): # 整数部のビット数を変更
        if (self.Type == 'V'):
            self.NumInte = Num
            self.bits = self.NumInte+self.NumDec+1
            print('整数部：%dbits, 小数部：%dbits' % (self.NumInte, self.NumDec))

    def Change_NumDec(self, Num): # 小数部のビット数を変更
        if (self.Type == 'V'):
            self.NumDec = Num
            self.bits = self.NumInte+self.NumDec+1
            print('整数部：%dbits, 小数部：%dbits' % (self.NumInte, self.NumDec))

    def Change_Type(self): # 問題のタイプを変更する
        if (self.Type == 'V'): # 連続変数 -> グラフ
            self.Type = 'G'
            self.NumInte = 0
            self.NumDec = 0
            self.bits = 1
            print('問題のタイプをグラフに変更しました.')
        else:                  # グラフ -> 連続変数
            self.Type = 'V'
            self.NumInte = 7
            self.NumDec = 4
            self.bits = self.NumInte+self.NumDec+1
            print('問題のタイプを連続変数に変更しました.')

    # DIM次元のデータセット'X'と正解のラベル'y'を作成する データ数はデフォルトで100個 
    def Set_DataSet(self, DIM=5, NumData=100, Filename=None):
        if Filename is None: # ファイル名に指定がない場合
            self.DIM = DIM
            self.NumData = NumData
            self.X = np.zeros((DIM, NumData))
            self.y = None
            answer = np.random.normal(0, scale=32, size=DIM)
            print(answer)
            
            for i in range(DIM):
                self.X[i,:] = np.random.normal(0, scale=1, size=NumData)
                if self.y is None:
                    self.y  = answer[i]*self.X[i,:]
                else:
                    self.y += answer[i]*self.X[i,:]
        else: # ファイル名に指定がある場合 .csvファイルに限る
            df = pd.read_csv(Filename)
            self.DIM = df.shape[1]-1
            self.NumData = df.shape[0]
            self.X = np.zeros((DIM, NumData))
            self.y = [NumData]
            for i in range(NumData):
                for j in range(DIM):
                    self.X[j][i] = df.iat[j][i]
                self.y[i] = df.iat[DIM+1][i]

    def Print_DataSet(self): # データセットの内容を表示する
        print('DIM：%d, NumData：%d\n'% (self.DIM, self.NumData))
        print(self.X)
        print(self.y)

    def Set_Matrix_G(self): # グラフ問題の場合、行列の各要素が頂点間の重みに対応する
        
        
    #def Set_Matrix_V(self): # 連続値を用いた最適化問題の場合、二変数間の積の係数が行列の要素に対応する

                
    #def Append_CostFunction_LS(self, ): # 最小二乗法をコスト関数に追加する

    #def Appned_CostFunction_l1(self, )

if __name__ == '__main__':
    import mglearn

    # X[DIM,NumData]:問題のデータセット, y[NumData]:問題のラベル, A[NumData]:問題の解
    p = Quantum_MonteCarlo()
    p.Set_DataSet()
    p.Print_DataSet()
    
    print('グラフ問題を作成する')
    DIM = 5 # 頂点の数を設定する     
