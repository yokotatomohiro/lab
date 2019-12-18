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

# 量子ビットのクラス
class Qbits():
    def set_qbits(self, DIM, bits):
        self.qbits     = np.zeros((DIM, bits), dtype=np.int0)
        self.qbits_np  = np.zeros(DIM*bits, dtype=np.int0)
        self.qbits_pre = np.zeros((DIM, bits), dtype=np.int0)
        self.qbits_pre_np = np.zeros(DIM*bits, dtype=np.int0)

    def qbit_inverse(self):
        random = randint(self.qbits.size)
        self.qbits[random] = 1 ^ self.qbits_pre[random]

    def qbits_inverse(self, Num): # ビット反転を行うビット数
        random_set = set()

        while len(random_set) < Num:
            random_set.add(randint(0, self.qbits.size))
            
        for i in random_set:
            self.qbits[i] = 1 ^ self.qbits_pre[i]
        
class Detale(): # 詳細な条件の設定
    def __init__(self):
        self.NumInte = 7     # 整数部のビット数
        self.NumDec  = 4     # 小数部のビット数
        self.bits    = 11    # 1変数あたりに使用するビット数
            
    def Change_NumInte(self, Num): # 整数部のビット数を変更
        self.NumInte = Num
        self.bits = self.NumInte+self.NumDec+1
        print('整数部：%dbits, 小数部：%dbits' % (self.NumInte, self.NumDec))

    def Change_NumDec(self, Num): # 小数部のビット数を変更
        self.NumDec = Num
        self.bits = self.NumInte+self.NumDec+1
        print('整数部：%dbits, 小数部：%dbits' % (self.NumInte, self.NumDec))

        
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

class Ising_Model(Qbits, Detale, DataSet): # イジングモデルを作成するクラス
    def Set_Ising_Model_Lasso(self, Lamnda=1, M=100):
        self.qbits(self.DIM*self.bits*3) # 量子ビットを設定する（次元数×1変数あたりのビット数×3）

        # 最小二乗法の部分（定数部を除く）
        # (DIM×3)×(DIM×3)の行列Jを作成する
        self.J = np.zeros((self.DIM*3, self.DIM*3)) # 変数はx,m_1,m_2の順
        J_sub = self.X**2
        for i in range(self.DIM): # Jの対角成分を埋める
            self.J[i*3][i*3] += np.sum(J_sub[i,:])
        # (DIM×3)のベクトルhを作成する
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
                
    
class Calculate_Cost(Ising_Model):
    def calculate_initial(self):
        self.cost = 0

        # 二体相互作用を計算
        for j in range(self.J.shape[1]):
            tmp = 0
            for i in range(j+1)
                DIM_i = i*self.bits+1
                DIM_j = j*self.bits+1
                bits = 0
                for k in range(self.bits-1):
                    for l in range(self.bits-1):
                        bit   = self.qbits[DIM_i+k]&self.qbits[DIM_j+l]
                        bits += bit * 2**(2*(self.NumInte-1)-k-l)
                sign = self.qbits[DIM_i-1]^self.qbits[DIM_j-1]
                if sign is 0:
                    tmp = J[i,j]*bits
                else:
                    tmp = -J[i,j]*bits
                self.cost += tmp
                
        # 一体相互作用を計算
        for i in range(self.h.shape[0]):
            tmp = 0
            for 
            
                
    def calculate_sub(self):


        
class Annealing(Detale, DataSet):
    def __init__(self):
        super().__init_()

    def Setup(self):
        self.qbits(self.bits*self.DIM) # bits*DIMの量子ビット列を作成する
        

    def annealing(self, )

# データセットおよび詳細な条件についての設定を行う
class Quantum_MonteCarlo():
    def __init__(self): # 各インスタンス変数の初期化
        self.J = None # 二体相互作用の係数
        self.h = None # 一体相互作用の係数


if __name__ == '__main__':
    import mglearn
    '''
    # X[DIM,NumData]:問題のデータセット, y[NumData]:問題のラベル, A[NumData]:問題の解
    model = DataSet()
    model.Make_DataSet(NumData=5)
    print(model.X)
    print(model.y)
    print('グラフ問題を作成する')
    DIM = 5 # 頂点の数を設定する  
    '''
    model = Detale()
    model.Change_NumInte(3)
    model.qbits_inverse(4)
    print(model.qbits)
    
