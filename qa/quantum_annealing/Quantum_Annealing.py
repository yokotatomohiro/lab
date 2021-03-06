# -*- coding:utf-8 -*-
import numpy as np
import random
import math

# パラメータの設定　初期値
ANN_STEP = 10 # アニーリング数
TEMPERATURE = 10 # 初期温度
Attenuation = 0.9999 # 減衰率

# 量子モンテカルロ法で利用するパラメータ
SIGMMA = 100 # 横磁場の強さの初期値
TROTTER= 4 # トロッタ数
MONTE_CARLO_STEP = TROTTER*2 # モンテカルロステップ数

# スピンを反転させる
def Spin_flip(spin, choise): # (spin[NumData], choise[1])
    if(spin[choise]==1):
        spin[choise] = 0
    else:
        spin[choise] = 1
        
# コストの計算を行う    
def Calculate_Cost(spin, matrix, vector): # (spin[NumData], matrix, vector)
    return (np.dot(np.dot(spin,matrix),spin)+np.dot(spin,vector))

# 横磁場を計算する
def Calculate_TMF(spin): # (spin[trotter,NumData])
    tmf = 0
    for i in range(spin.shape[0]-1):
        tmf += sum(4**(spin[i,:]*spin[i+1,:])-2**(spin[i,:]+spin[i+1,:])+1)
    return tmf

# spinの初期化を行う
def Initialization(spin): # (spin[trotter,NumData])
    for i in range(spin.shape[0]):
        for j in range(spin.shape[1]):
            spin[i,j] = random.randint(0,1)

# 横磁場の係数を計算する
def Transverse_magnetic_field(trotter, sigmma, temperature):
    return (-(temperature/2)*math.log(np.cosh(sigmma/(trotter*temperature))/np.sinh(sigmma/(trotter*temperature))))

# 各トロッタ次元のコストを計算し，最小となる次元のコストを返却する
def Return_min_cost(spin, matrix, vector): # (spin[tortter,NumData])
    min_trotter_dim = 0
    min_cost = Calculate_Cost(spin[0,:], matrix, vector)
    for i in range(1,spin.shape[0]):
        tmp = Calculate_Cost(spin[i,:], matrix, vector)
        if min_cost>tmp:
            min_cost = tmp
            min_trotter_dim = i
    return min_cost

def Return_min_spin(spin, matrix, vector):
    min_trotter_dim = 0
    min_cost = Calculate_Cost(spin[0,:], matrix, vector)
    for i in range(1,spin.shape[0]):
        tmp = Calculate_Cost(spin[i,:], matrix, vector)
        if min_cost>tmp:
            min_cost = tmp
            min_trotter_dim = i
    return spin[i,:]

class qmc():
    def __init__(self, MATRIX, VECTOR):
        self.matrix = MATRIX
        self.vector = VECTOR
        
    def Quantum_Annealing(self, trotter=TROTTER, sigmma=SIGMMA, att=Attenuation, ann_step=ANN_STEP, temperature=TEMPERATURE):
        monte_step = trotter*2
        # (self, トロッター数, モンテカルロ数, 横磁場の強さ, 減衰率, アニーリング数, 有効温度)
        self.spin = np.zeros((trotter, len(self.vector)), dtype=np.int)
        self.cost_point = np.array([])

        # スピンの初期化
        Initialization(self.spin)
        coeff = 0

        # アニーリングを行う
        choise = np.zeros(2, dtype=np.int) # 反転させる位置を決定[TROTTERの位置,SPINの位置]
        for i in range(ANN_STEP): # アニーリング数
            coeff = Transverse_magnetic_field(trotter, sigmma, temperature)
            for j in range(2*len(self.vector)): # 一回のイテレーションでのビット反転数
                for k in range(monte_step): # モンテカルロ数
                    choise = [random.randint(0,trotter-1), random.randint(0,len(self.vector)-1)]
                    
                    cost_a = Calculate_Cost(self.spin[choise[0],:], self.matrix, self.vector)
                    tmf_a = Calculate_TMF(self.spin)

                    Spin_flip(self.spin[choise[0],:],choise[1])

                    cost_diff = (Calculate_Cost(self.spin[choise[0],:], self.matrix, self.vector)-cost_a)/trotter+coeff*(Calculate_TMF(self.spin)-tmf_a)

                    if cost_diff<0: # 受理する
                        pass
                    elif math.exp(-cost_diff/temperature)>random.random(): # 受理する
                        pass
                    else:
                        Spin_flip(self.spin[choise[0],:],choise[1]) # 拒否する
            sigmma *= att # パラメータを更新する
            self.cost_point = np.append(self.cost_point,(Return_min_cost(self.spin, self.matrix, self.vector)))

        self.result_spin = Return_min_spin(self.spin, self.matrix, self.vector)

                
