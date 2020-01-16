# -*- coding: utf-8 -*-

import numpy as np
from numpy.random import *

class Qbits():
    def __init__(self, Num):
        self.size = Num # 量子ビット数の設定を行う
        self.qbits     = 0
        self.qbits_tmp = 0 # ビット反転後のビット列
        self.qbits_min = 0 # コストが最小となるビット列
        self.cost = 0
        self.cost_tmp = 0
        self.cost_min = 0
        self.mask = None

    def Set_mask(self, array):
        self.mask = (1 << self.size)-1
        for i in array:
            self.mask = self.mask ^ (1 << i)

    def Inverse_qbits(self): # ビット反転を行う
        self.qbits_tmp = self.qbits ^ (1 << randint(self.size))

        if self.mask is not None: # ビット反転に指定がある場合
            self.qbits_tmp = self.qbits_tmp & self.mask
            
    def Flip_qbits(self): # フリップを行う
        self.qbits = self.qbits_tmp
        self.cost = self.cost_tmp

    def Update_qbits_min(self): # 最小のビット列を更新する
        self.qbits_min = self.qbits
        self.cost_min = self.cost
'''
class Annealing():
    def __init__(self):
        # 一変数あたりに使用する量子ビット数の指定
        self.NumInte = 7
        self.NumDec  = 4
        self.bits    = self.NumInte+self.NumDec+1

        # 問題の設定を行う
        self.DIM     = 5
        self.NumData = 100
        self.X       = None
        self.y       = None
        self.answer  = None

        # QUBO形式の設定を行う
        self.J       = None
        self.h       = None
        self.J_qbits = None
        self.h_qbits = None
        self.array   = [] # ビット反転をさせない位置を格納

        # アニーリングのスケジュール
        self.T = 100
        self.alpha = 0.999
        self.iteration = 100

    # 一変数あたりのビット数を変更する
    def Change_bits(self, NumInte=None, NumDec=None):
        if NumInte is not None:
            self.NumInte = NumInte
        if NumDec is not None:
            self.NumDec = NumDec
        self.bits = self.NumInte+self.NumDec+1

    # 生成するデータセットの次元とデータ数を変更する
    def Change_detale(self, DIM=None, NumData=None):
        if DIM is not None:
            self.DIM = DIM
        if NumData is not None:
            self.NumData = NumData

    # データセットを作成する
    def Make_dataset(self, scale=None, noise=None):
        self.X = np.random.normal(0,1,(self.DIM,self.NumData))
        self.y = np.zeros(self.NumData)
        
        if scale is None:
            scale = 2**(self.NumInte-1)
        self.answer = np.random.normal(0,scale,self.NumData)

        for i in range(self.DIM):
            self.y += self.answer[i]*self.X[i,:]

        if noise is not None:
            self.y += np.random.normal(0,noise,self.NumData)

    # lassoのコストを設定する
    def Set_lasso(self, Lambda=1, M=100):
        size = self.DIM*3

        self.J = np.zeros((size,size))
        self.h = np.zeros(size)

        J_sub = self.X**2

        # 最小二乗法のコスト関数を設定 yは定数なので除く
        for i in range(self.DIM):
            self.J[i*3] += np.sum(J_sub[i,:])
            self.h[i*3] += -2*np.sum(self.X[i,:]*self.y)

        # l1ノルムのコスト関数
        for i in range(self.DIM):
            self.h[i*3+1] += Lambda
            self.h[i*3+2] += Lambda

        # l1ノルムのペナルティ項
        for i in range(self.DIM):
            for j in range(3):
                self.J[i*3+j,i*3+j] += M*Lambda
            self.J[i*3  ,i*3+1] +=  2*M*Lambda
            self.J[i*3  ,i*3+2] +=  2*M*Lambda
            self.J[i*3+1,i*3+2] += -2*M*Lambda
            
        # 連続値 -> 離散値 に変換する 
        self.J_qbits = np.zeros((size*self.bits,size*self.bits))
        self.h_qbits = np.zeros(size*self.bits)

        for i in range(size):
            for j in range(size):
                if self.J[i,j] != 0:
                    self.J_qbits[i*self.bits,j*self.bits] = self.J[i,j]*2**(self.NumInte*2)
                    for k in range(1,self.bits):
                        for l in range(1,self.bits):
                            self.J_qbits[i*self.bits+k,j*self.bits+l] = self.J[i,j]*2**((self.NumInte-k)+(self.NumInte-l))
                        self.J_qbits[i*self.bits,j*self.bits+k] = -self.J[i,j]*(2**(self.NumInte*2-k))
                        self.J_qbits[i*self.bits+k,j*self.bits] = -self.J[i,j]*(2**(self.NumInte*2-k))
            if self.h[i] != 0:
                self.h_qbits[i*self.bits] = -self.h[i]*(2**(self.NumInte))
                for j in range(1,self.bits):
                    self.h_qbits[i*self.bits+k] = self.h[i]*(2**(self.NumInte-j))

        for i in range(size):
            self.array.append((3*i*self.bits)+self.bits)
            self.array.append((3*i*self.bits)+self.bits*2)

    def Calculate(self, qbits):
        cost = 0
        place = []
        for i in range(self.h_qbits.size):
            if((qbits >> i) & 1) == 1:
                place.append(i)
                
        J_sub = []
        for i in place:
            for j in place:
                if self.J_qbits[i,j] != 0:
                    J_sub.append(self.J_qbits[i,j])
        cost += np.sum(J_sub)

        h_sub = []
        for i in place:
            if self.h_qbits[i] != 0:
                h_sub.append(self.h_qbits[i])
        cost += np.sum(h_sub)

        return cost

    def Change_anneal(self, T=None, alpha=None, iteration=None):
        if T is not None:
            self.T = T
        if alpha is not None:
            self.alpha = alpha
        if iteration is not None:
            self.iteration = iteratio

    def Update_anneal(self):
        self.T *= self.alpha

    def flip_check(self, diff_cost):
        if diff_cost < 0:
            return True
        else:
            if rand() < np.exp(diff_cost/self.T):
                return False
            else:
                return True

    def annealing(self):
        qbits = Qbits(self.h_qbits.size)
        qbits.Set_mask(self.array)
        
        for i in range(self.iteration):
            self.Update_anneal()
            qbits.Inverse_qbits()
            qbits.cost_tmp = self.Calculate(qbits.qbits_tmp)

            diff = qbits.cost_tmp - qbits.cost

            if self.flip_check(diff) is True:
                qbits.Flip_qbits()
            if qbits.cost_min > qbits.cost:
                qbits.Update_qbits_min()

        print(bin(qbits.qbits_min))
'''
                
if __name__ == '__main__':
    a = Qbits(5)
    print(bin(a.qbits))
    for i in range(5):
        a.Inverse_qbits()
        a.Flip_qbits()
        print(bin(a.qbits))
