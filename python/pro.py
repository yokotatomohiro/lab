# -*- coding: utf-8 -*-

import numpy as np
from numpy.random import *
import pandas as pd
import sys
import math

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

max = 1000

# 量子ビットのクラス
class Qbits():
    def set_qbits(self, DIM):
        self.qbits     = np.zeros(DIM, dtype=np.int0) # ビット反転後の量子ビット値
        self.qbits_pre = np.zeros(DIM, dtype=np.int0) # ビット反転前の量子ビット値
        self.qbits_min = np.zeros(DIM, dtype=np.int0) # 最小となる量子ビット値

    def qbit_inverse(self): # １つの量子ビットをビット反転させる
        random = randint(self.qbits.size)
        self.qbits[random] = 1 ^ self.qbits_pre[random]
        print(random)

    def qbits_inverse(self, Num): # 複数の量子ビットをビット反転させる
        random_set = set()

        while len(random_set) < Num:
            random_set.add(randint(0, self.qbits.size))
            
        for i in random_set:
            self.qbits[i] = 1 ^ self.qbits_pre[i]

    def qbits_reset(self): # 
        self.qbits = self.qbits_pre

    def qbits_update_pre(self):
        self.qbits_pre = self.qbits

    def qbits_update_minimum(self):
        self.qbits_minimum = self.qbits
        
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

class Print_qbits(Qbits, Detale):
    def qbits_to_num(self):
        Num = self.qbits_min.size # 量子ビットの数
        DIM = int(Num/self.bits) # 変数の数
        num = []
        print(self.qbits_min)
        
        for i in range(DIM):
            tmp = 0
            count = 1
            for j in range(self.NumInte-1, self.NumDec+1, -1):
                tmp += self.qbits[i*self.qbits+count]*2**j
                count += 1
            if self.qbits_min[i*self.qbits] is 1:
                tmp = -tmp
            num.append(tmp)

        print(num)
    
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
        self.y = np.zeros(NumData)
        answer = np.random.normal(0, scale=32, size=DIM)
        print(answer)
        
        for i in range(DIM):
            self.X[i,:] = np.random.normal(0, scale=1, size=NumData)
            self.y = answer[i]*self.X[i,:]
            if Noise is not None:
                self.y += np.random.noise(0, scale=Noise, size=NumData)

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

class Ising_Model(Qbits, Detale, DataSet): # イジングモデルを作成するクラス（まだ二進数展開をしない）
    def __init__(self):
        super().__init__()
        
    def Set_Ising_Model_Lasso(self, Lambda=1, M=100): # コスト関数にlassoを設定する
        self.set_qbits(self.DIM*self.bits*3) # 量子ビットを設定する（次元数×1変数あたりのビット数×3）

        # 最小二乗法の部分（定数部を除く）
        # (DIM×3)×(DIM×3)の行列Jを作成する
        self.J = np.full((self.DIM*3, self.DIM*3), max) # 変数はx,m_1,m_2の順
        J_sub = self.X**2
        # (DIM×3)のベクトルhを作成する
        self.h = np.full(self.DIM*3, max) 
        for i in range(self.DIM):
            self.h[i*3] += np.sum(J_sub[i,:])-2*np.sum(self.X[i,:]*self.y) # hの成分を埋める
        
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
    def calculate_initial(self): # 初回のコストを計算する
        self.cost = 0
        self.cost_minimum = 0
        
        # 二体相互作用を計算
        J_sub = []
        print(self.J.shape[0], self.J.shape[1])
        for i in range(self.J.shape[0]):
            for j in range(self.J.shape[1]):
                #if self.J[i][j] is not max:
                coeff = self.J[i][j]
                sign = self.qbits[i*self.bits] ^ self.qbits[j*self.bits]
                count_k = 1
                for k in range(self.NumInte-1, -self.NumDec+1, -1):
                    count_l = 1
                    for l in range(self.NumInte-1, -self.NumDec+1, -1):
                        if sign is 0:
                            J_sub.append( coeff*(self.qbits[i*self.bits+count_k] ^ self.qbits[j*self.bits+count_l])*2**(k+l))
                        else:
                            J_sub.append(-coeff*(self.qbits[i*self.bits+count_k] ^ self.qbits[j*self.bits+count_l])*2**(k+l))
                        count_l += 1
                    count_k += 1
        self.cost += np.sum(J_sub)
        
        # 一体相互作用を計算
        h_sub = []
        for i in range(self.h.size):
            #if self.h[i] is not 0:
                coeff = self.h[i]
                count_j = 1
                for j in range(self.NumInte-1, -self.NumDec+1, -1):
                    if self.qbits[i*self.bits] is 0:
                        h_sub.append( coeff*(self.qbits[i*self.bits+count_j])*2**j)
                    else:
                        h_sub.append(-coeff*(self.qbits[i*self.bits+count_j])*2**j)
                    count_j += 1  
        self.cost += np.sum(h_sub)
        self.cost_miminum = self.cost
                
    def calculate_diff(self): # ビット反転したことによるコストの差分を計算する
        cost = 0      # ビット反転前のコスト
        cost_pre = 0  # ビット反転後のコスト
        place = set() # ビット反転した変数の位置を格納する
        
        for i in range(self.qbits.size):
            if (self.qbits[i] ^ self.qbits_pre[i]) is 1:
                place.add(i // self.bits)

        J_sub = []
        J_pre_sub = []
        for i in place:
            for j in range(self.J.shape[0]):
                #if J[i][j] is not 0:
                    coeff = J[i][j]
                    sign     = self.qbits[i*self.bits] ^ self.qbits[j*self.bits]
                    sign_pre = self.qbits_pre[i*self.bits] ^ self.qbits_pre[j*self.bits]
                    count_k = 1
                    for k in range(self.NumInte-1, -self.NumDec+1, -1):
                        count_l = 1
                        for l in range(self.NumInte-1, -self.NumDec+1, -1):
                            if sign is 0:
                                J_sub.append( coeff*(self.qbits[i*self.bits+count_k] ^ self.qbits[j*self.bits+count_l])*2**(k+l))
                            else:
                                J_sub.append(-coeff*(self.qbits[i*self.bits+count_k] ^ self.qbits[j*self.bits+count_l])*2**(k+l))
                            if sign_pre is 0:
                                J_pre_sub.append( coeff*(self.qbits_pre[i*self.bits+count_k] ^ self.qbits_pre[j*self.bits+count_l])*2**(k+l))
                            else:
                                J_pre_sub.append(-coeff*(self.qbits_pre[i*self.bits+count_k] ^ self.qbits_pre[j*self.bits+count_l])*2**(k+l))
                            count_l += 1
                        count_k += 1
        cost     += np.sum(J_sub)
        cost_pre += np.sum(J_pre_sub)

        h_sub = []
        h_pre_sub = []
        for i in place:
            coeff = h[i]
            count_j = 1
            for j in range(self.NumInte-1, -self.NumDec+1, -1):
                if self.qbits[i*self.qbits] is 0:
                    h_sub.append( coeff*(self.qbits[i*self.bits+count_i])*2**j)
                else:
                    h_sub.append(-coeff*(sefl.qbits[i*self.bits+count_i])*2**j)
                if self.qbits[i*self.qbits] is 0:
                    h_sub.append( coeff*(self.qbits_pre[i*self.bits+count_i])*2**j)
                else:
                    h_sub.append(-coeff*(self.qbits_pre[i*self.bits+count_i])*2**j)
                count_i += 1
        cost     += np.sum(h_sub)
        cost_pre += np.sum(h_pre_sub)

        return (cost_pre-cost)
        
class Annealing(Calculate_Cost, Ising_Model, Print_qbits):
    def __init__(self):
        super().__init__()
        self.T = 50
        self.alpha = 0.9995
        self.iteration = 10000

    def Change_T(self, T=50):
        self.T = T
        
    def Change_alpha(self, alpha=0.9995):
        self.alpha = alpha

    def Change_iteration(self, iteration=10000):
        self.iteration = iteration

    def annealing(self):
        self.calculate_initial()
        self.qbits_update_minimum()
        
        for i in range(self.iteration):
            self.T += self.alpha
            self.qbit_inverse()
            print('self.qbits=', self.qbits)
            print('self.qbits_pre=')
            #print(self.qbits)
            #self.qbits_inverse(5)
            diff = self.calculate_diff()
            print(diff)
            if (diff < 0.0):
                self.qbits_update_pre()
                print('flip')
                self.cost += diff
            elif (rand() < math.exp(-diff/self.T)):
                self.qbits_update_pre()
                print('flip')
                self.cost += diff
            else:
                self.qbits_reset()
                print('not flip')

            if (self.cost < self.cost_minimum):
                self.cost_minimum = self.cost
                self.qbits_update_minimum()

        self.qbits_to_num()
        
            

        
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

    p = Annealing()
    p.Make_DataSet(DIM=1) # データセットの設定を行う
    p.Change_NumInte(2)
    p.Change_NumDec(2)
    p.Change_iteration(3)
    p.Set_Ising_Model_Lasso() # 問題の設定を行う
    print(p.J)
    print(p.h)
    p.annealing() # アニーリングを行う
    
