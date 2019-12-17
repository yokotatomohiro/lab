# -*- coding: utf-8 -*-

import numpy as np
from numpy.random import *
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
"""
class Detale(): # 詳細な条件の設定
    def __init__(self):
        self.NumInte = 7     # 整数部のビット数
        self.NumDec  = 4     # 小数部のビット数
        self.bits    = 11    # 1変数あたりに使用するビット数
        self.Type    = 'V'   # 問題のタイプを表す 'V':'連続変数', 'G':'グラフ'
            
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
            self.bits = 11
            print('問題のタイプを連続変数に変更しました.')

class DataSet():
    def __init__(self):
        self.DIM = None
        self.NumData = None
        self.X = None
        self.y = None
        
    def Make_DataSet(self, DIM=5, NumData=100, Noise=None):
        self.DIM = DIM
        self.NumData = NumData
        self.X = np.zeros((DIM, NumData))
        answer = np.random.normal(0, scale=32, size=DIM)
        print(answer)
        
        for i in range(DIM):
            self.X[i,:] = np.random.normal(0, scale=1, size=NumData)
            if self.y is None:
                self.y = answer[i]*self.X[i,:]
                if Noise is not None:
                    self.y += np.random.noise(0, scale=Noise, size=NumData)
            else:
                self.y += answer[i]*self.X[i,:]

    def Import_DataSet(self, filename):
        df = pd.read_csv(Filename)
        self.DIM = df.shape[1]-1
        self.NumData = df.shape[0]
        self.X = np.zeros((DIM, NumData))
        self.y = [NumData]
        for i in range(NumData):
            for j in range(DIM):
                self.X[j][i] = df.iat[j][i]
            self.y[i] = df.iat[DIM+1][i]
    

# データセットおよび詳細な条件についての設定を行う
class Quantum_MonteCarlo():
    def __init__(self): # 各インスタンス変数の初期化
        self.J = None # 二体相互作用の係数
        self.h = None # 一体相互作用の係数

    def Ising_Model_Lasso(self, Lambda=1, M=100):
        # 最小二乗法の部分
        if self.J is None:
            self.J = np.zeros((self.DIM*3, self.DIM*3)) # 変数はx,m_1,m_2の順
        J_sub = self.X**2
        for i in range(self.DIM): # Jの対角成分を埋める
            self.J[i*3][i*3] += np.sum(J_sub[i,:])
        if self.h is None:
            self.h = np.zeros(self.DIM*3) 
        for i in range(self.DIM):
            self.h[i] += -2*np.sum(self.X[i,:]*self.y) # hの成分を埋める
        
        # l-1normの部分
        for i in range(self.DIM): # 目的関数の部分
            self.h[i*3]   += self.h[i]
            self.h[i*3+1] += Lambda
            self.h[i*3+2] += Lambda
        for i in range(self.DIM): # 制約項の部分
            for j in range(3): 
                self.J[i*3+j][i*3+j] += M*Lambda
            self.J[i*3][i*3+1] += M*2*Lambda # m*z_1
            self.J[i*3][i*3+2] += M*-2*Lambda # m*z_2
            self.J[i*3+1][i*3+2] += M*-2*Lambda # z_1*z_2
                
# 量子ビットのクラス
class Qbits():
    def __init__(self, Num): # Num:量子ビット数
        self.qbits = np.zeros(Num, dtype=np.int0)

    def qbit_inverse(self):
        random = randint(self.qbits.size)
        self.qbits[random] = 1 ^ self.qbits[random]

    def qbits_inverse(self, Num): # ビット反転を行うビット数
        random_set = set()

        while len(random_set) < Num:
            random_set.add(randint(0, self.qbits.size))
            
        for i in random_set:
            self.qbits[i] = 1 ^ self.qbits[i]
        
                
if __name__ == '__main__':
    import mglearn
    
    # X[DIM,NumData]:問題のデータセット, y[NumData]:問題のラベル, A[NumData]:問題の解
    model = DataSet()
    model.Make_DataSet(NumData=5)
    print(model.X)
    print(model.y)
    print('グラフ問題を作成する')
    DIM = 5 # 頂点の数を設定する  

    
