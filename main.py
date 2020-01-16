# -*- coding: utf-8 -*-

import numpy as np
from numpy.random import *

class Detale():
    def __init__(self):
        # 一変数あたりに使用するビット数の指定
        self.NumInte = 7 # 整数部のビット数
        self.NumDec  = 4 # 小数部のビット数
        self.bits    = self.NumInte+self.NumDec+1 # 符号部を含めた全体のビット数

        # データセットの詳細についてい
        self.DIM     = 5    # データセットの次元数
        self.NumData = 100  # データセットのデータ数
        self.X       = None # 計画行列
        self.y       = None # 
        self.answer  = None # 解

        self.J       = None # 連続値の二体相互作用対応表
        self.h       = None # 連続値の一体相互作用対応表
        self.J_qbits = None # 離散値の二体相互作用対応表
        self.h_qbits = None # 離散値の一体相互作用対応表
        self.mask    = 0    # 離散値にいてビット反転させてはいけない位置のマスクビット列

    def change_bits(self, NumInte=None, NumDec=None):
        if NumInte is not None:
            self.NumInte = NumInte
        if NumDec is not None:
            self.NumDec = NumDec
        self.bits = self.NumInte+self.NumDec+1

    def change_dataset(self, DIM=None, NumData=None):
        if DIM is not None:
            self.DIM = DIM
        if NumData is not None:
            self.NumData = NumData

    def make_dataset(self, scale=None, noise=0):
        # データセットの初期化
        self.X = np.random.normal(0,1,(self.DIM,self.NumData))
        self.y = np.zeros(self.NumData)

        if scale is None:
            scale = 2**(self.NumInte-1)
        self.answer = np.random.normal(0,scale,self.DIM)

        for i in range(self.DIM):
            self.y += self.answer[i]*self.X[i,:]

        self.y += np.random.normal(0,noise,self.NumData)

    # lassoのコスト関数を設定する
    def set_lasso(self, l=1, m=100):
        size = self.DIM*3

        self.J = np.zeros((size,size))
        self.h = np.zeros(size)

        J_sub = self.X**2

        # 最小二乗法のコスト関数を設定 yは定数なので除く
        for i in range(self.DIM):
            self.J[i*3,i*3] += np.sum(J_sub[i,:])
            self.h[i*3    ] += -2*np.sum(self.X[i,:]*self.y)

        # l1ノルムのコスト関数を設定
        for i in range(self.DIM):
            self.h[i*3+1] += l
            self.h[i*3+2] += l

        # l1ノルムのペナルティ項を設定
        for i in range(self.DIM):
            for j in range(3):
                self.J[i*3+j,i*3+j] += m*l
            self.J[i*3  ,i*3+1] += 2*m*l
            self.J[i*3  ,i*3+2] += -2*m*l
            self.J[i*3+1,i*3+2] += -2*m*l

        # 連続値 -> 離散値に変換する
        size_qbits = size*self.bits
        self.J_qbits = np.zeros((size_qbits,size_qbits))
        self.h_qbits = np.zeros(size_qbits)

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

        # ビット反転させてはいけない位置を格納
        self.mask = (1 << size_qbits)-1
        for i in range(self.DIM):
            self.mask = self.mask ^ (1<<((3*i+1)*self.bits))
            self.mask = self.mask ^ (1<<((3*i+2)*self.bits))

# ビット反転を行う
def inverse_qbits(qbits, size, mask=None, Num=1): # qbits:量子ビット列, size:量子ビット数, mask:マスク, Num:ビット反転を行う数
    tmp = qbits
    if mask is None:
        for i in range(Num):
            tmp = tmp ^ (1 << randint(size))
    else:
        while(1): # 少なくとも１つはビット反転する
            for i in range(Num):
                tmp = tmp ^ (1 << randint(size))
            if mask is not None:
                tmp = tmp & mask
            if tmp != qbits:
                break
        return tmp

# コストの計算を行う
def calculate_discrete(qbits, J, h): # qbits:量子ビット列(int型), J:[ビット数,ビット数], h:[ビット数]
    cost = 0
    size = h.size
    place = []
    for i in range(size):
        if ((qbits>>i) & 1) != 0:
            place.append(i)

    sub = []
    for i in place:
        for j in place:
            if J[i,j] != 0:
                sub.append(J[i,j])
        if h[i] != 0:
            sub.append(h[i])

    return np.sum(sub)

def Annealing(detale, iteration=100, T=10000, alpha=0.995):
    size = detale.h_qbits.size
    qbits = 0
    qbits = inverse_qbits(qbits, size, detale.mask, size)
    cost = calculate_discrete(qbits, detale.J_qbits, detale.h_qbits)
    min_qbits = qbits
    min_cost = cost

    for i in range(iteration):
        T *= alpha
        tmp = inverse_qbits(qbits, size, detale.mask)
        tmp_cost = calculate_discrete(tmp, detale.J_qbits, detale.h_qbits)
        diff = tmp_cost - cost
        if diff < 0:
            qbits = tmp
            cost = tmp_cost
        else:
            if rand() < np.exp(diff/T):
                pass
            else:
                qbits = tmp
                cost = tmp_cost

        if min_cost > cost:
            min_qbits = qbits
            min_cost = cost


    print(min_cost)
    print(bin(min_qbits))

def print_discrete(detale, qbits): # 離散値->連続値にして表示する
    Num = detale.h_space/(detale.bits*detale.DIM)
    
    for i in range(detale.DIM):
        for j in range(Num):
            value = 0
            if ((qbits >> i*detale.bits*Num) & 1) == 1
            value = -1<<detale.NumInt
            

if __name__ == '__main__':
    a = Detale()
    a.change_bits(1,1)
    a.make_dataset()
    a.set_lasso()
    
    print(a.answer)

    Annealing(a)

