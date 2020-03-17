
# -*- coding:utf-8 -*-
import time
import math
import numpy as np
import pandas as pd
from scipy.stats import zscore
import os
import random
import matplotlib.pyplot as plt

"""
量子モンテカルロシミュレーションのエネルギー関数
E = Σ^{N}_{i,j=1}{Σ^{m}_{k=1}{J_{ij}σ_{i,k}σ_{j,k}}}
   +Σ^{N}_{i=1}{Σ^{m}_{k=1}{h_{i=1}σ_{i,k}}}
   +\frac{m}{2β}log{coth{\frac{βΓ}{m}}}Σ^{N}_{i=1}{Σ^{m}_{k=1}{σ_{i,k}σ_{i,k+1}}}

量子モンテカルロシミュレーションの手順
1. 逆温度βと横磁場の強さΓを決める．
2. m個の層のそれぞれのN個のスピンの配置をランダムに決める（初期化）．
3. エネルギーEを計算する．ランダムに層とその層の中のスピンを選びスピンの向きを変える．エネルギーを計算しE'とする．
　　確率min(1,exp(-βΔE))でスピンの向きの変更を許容し，状態変更をコミットする．
4. 決められた回数(モンテカルロステップ数)だけ3.の処理を繰り返しエネルギーを下げていく．
5. 横磁場の強さΓを小さくする．
6. 4.と5.をある決められた回数(アニーリング数)行う 
7. 最後に第三項をを除いた各層のエネルギーを求め，一番エネルギーの低い層の配置を解として採用する．
"""

# パラメータの設定
#TROTTER_DIM = int(input("トロッター次元数: m = ")) 
#ANN_PARA = float(input("初期パラメータ: Γ_init = "))
#ANN_STEP = int(input("アニーリング数: ANN = "))
#MC_STEP = int(input("モンテカルロステップ: MC = "))
#BETA = float(input("初期逆温度: β = "))
#REDUCE_PARA = 0.99 # 減衰率

# データセットの内容を取得する .csvファイルを読み込む
FILE_NAME = str(input(".csvファイル名 ："))
FILE_DATA = pd.read_csv(FILE_NAME, index_col=0)
FILE_columns_X_NAME = list(FILE_DATA.columns[0:-1])
FILE_columns_Y_NAME = list(FILE_DATA.columns[-1])
DATA_X = FILE_DATA.iloc[:,:-1].values
DATA_X = zscore(DATA_X) # 列ごとに標準化する
DATA_Y = FILE_DATA.iloc[:,-1].values
DIM = DATA_X.shape[1]
NumData = DATA_X.shape[0]

# 計算で利用するデータを作成する
# DATA_TwoBody:(次元×次元)行列，DATA_OneBody:(次元)ベクトル，DATA_Const:定数
# 初期化
DATA_TwoBody = np.zeros((DIM,DIM)) # 二体相互作用
DATA_OneBody = np.zeros(DIM) # 一体相互作用
DATA_Const = 0

# コスト関数（最小二乗法で利用） ||Xβ-y||^2 定数部（y^2）を含む場合
for i in range(0,DIM):
    DATA_TwoBody[i,i] = sum(np.square(DATA_X[:,i]))
    for j in range(0,i):
        DATA_TwoBody[i,j] = 2*sum(DATA_X[:,i]*DATA_X[:,j])
    DATA_OneBody[i] = -2*sum(DATA_X[:,i]*DATA_Y[:])
DATA_Const = sum(np.square(DATA_Y))

LAMBDA = int(input("l1-normの正則化パラメータ：λ = "))
M_penalty = int(input("l1-normのペナルティ：M = "))

# コスト関数（l-1normで利用） λ(z_1+z_2+M(-β-z_1+z_2)^2)

NumInte = int(input("実数部のビット数：NumInte = "))
NumDec = int(input("小数部のビット数：NumDec = "))

print("%f ≦　β　≦ %f" %  (-pow(2,NumInte), pow(2,NumInte)-pow(2,-NumDec)))
print("    0     ≦　z　≦ %f" %  (pow(2,NumInte)-pow(2,-NumDec)))

MATRIX_BB = np.zeros((NumInte+NumDec+1,NumInte+NumDec+1))
MATRIX_BZ = np.zeros((NumInte+NumDec  ,NumInte+NumDec+1))
MATRIX_ZZ = np.zeros((NumInte+NumDec  ,NumInte+NumDec))
BECTOR_B  = np.zeros(NumInte+NumDec+1)
BECTOR_Z  = np.zeros(NumInte+NumDec)

# 離散値に対応する行列とベクトル β×β，β×z，z×z, β, z
for i in range(NumInte+NumDec+1):
    for j in range(NumInte+NumDec+1):
        MATRIX_BB[i,j] = pow(2,(NumInte-i)+(NumInte-j))
for i in range(1,NumInte+NumDec+1):
    MATRIX_BB[0,i] = -MATRIX_BB[0,i]
    MATRIX_BB[i,0] = -MATRIX_BB[i,0]

for i in range(NumInte+NumDec):
    for j in range(NumInte+NumDec+1):
        MATRIX_BZ[i,j] = pow(2,(NumInte-i-1)+(NumInte-j))
MATRIX_BZ[:,0] = -MATRIX_BZ[:,0]

for i in range(NumInte+NumDec):
    for j in range(NumInte+NumDec):
        MATRIX_ZZ[i,j] = pow(2,(NumInte-i-1)+(NumInte-j-1))

for i in range(NumInte+NumDec+1):
    BECTOR_B[i] = pow(2,(NumInte-i))
BECTOR_B[0] = -BECTOR_B[0]

for i in range(NumInte+NumDec):
    BECTOR_Z[i] = pow(2,(NumInte-i-1))
        
# 変数の並び順
# β_0, β_1,..., β_n, z_1:0, z_1:1,..., z_1:n, z_2:0, z_2:1,...,z_2:n 
DATA_TwoBody_Qubits = np.zeros((DIM*(3*(NumInte+NumDec)+1),DIM*(3*(NumInte+NumDec)+1)))
DATA_OneBody_Qubits = np.zeros(DIM*(3*(NumInte+NumDec)+1))
BtoZ1 = DIM*(NumInte+NumDec+1)
Z1toZ2 = DIM*(2*(NumInte+NumDec)+1)

"""
DATA_TwoBody_Qubitsの配列
βのビット数：DIM*(NumInt+NumDec+1)
z1,z2のビット数：DIM*(NumInte+NumDec)

            β         z1        z2
      　__________  _________  _________
       |         | |        | |        |
       |         | |        | |        |
 β     |    BB   | |        | |        |
       |         | |        | |        |
       |_________| |________| |________|
BtoZ1   __________  _________  _________
       |         | |        | |        |
 z1    |   Z1B   | |  Z1Z1  | |        |
       |         | |        | |        |
       |_________| |________| |________|
Z1toZ2　__________  _________  _________
       |         | |        | |        | 
 z2    |   Z2B   | |  Z2Z1  | |  Z2Z2  |
       |         | |        | |        | 
       |_________| |________| |________|

"""

for i in range(DIM):
    DATA_TwoBody[i,i] += LAMBDA*M_penalty
# 二体相互作用の組み込みを行う
# BBの組み込みを行う
for i in range(DIM):
    for j in range(i+1):
        coeff = DATA_TwoBody[i,j]*MATRIX_BB
        for k in range(MATRIX_BB.shape[0]):
            for l in range(MATRIX_BB.shape[1]):
                DATA_TwoBody_Qubits[i*MATRIX_BB.shape[0]+k,j*MATRIX_BB.shape[1]+l] = coeff[k,l]

# Z1BとZ2Bの組み込みを行う
coeff = 2*LAMBDA*M_penalty*MATRIX_BZ
for i in range(DIM):
    for k in range(MATRIX_BZ.shape[0]):
        for l in range(MATRIX_BZ.shape[1]):
            DATA_TwoBody_Qubits[i*MATRIX_BZ.shape[0]+k+BtoZ1,i*MATRIX_BZ.shape[1]+l] = coeff[k,l]
            DATA_TwoBody_Qubits[i*MATRIX_BZ.shape[0]+k+Z1toZ2,i*MATRIX_BZ.shape[1]+l] = -coeff[k,l]

# Z2Z1の組み込みを行う
coeff = -2*LAMBDA*M_penalty*MATRIX_ZZ
for i in range(DIM):
    for k in range(MATRIX_ZZ.shape[0]):
        for l in range(MATRIX_ZZ.shape[1]):
            DATA_TwoBody_Qubits[i*MATRIX_ZZ.shape[0]+k+Z1toZ2,i*MATRIX_ZZ.shape[1]+l+BtoZ1] = coeff[k,l]

# Z1Z1とZ2Z2の組み込みを行う
coeff = LAMBDA*M_penalty*MATRIX_ZZ
for i in range(DIM):
    for k in range(MATRIX_ZZ.shape[0]):
        for l in range(MATRIX_ZZ.shape[1]):
            DATA_TwoBody_Qubits[i*MATRIX_ZZ.shape[0]+k+BtoZ1,i*MATRIX_ZZ.shape[1]+l+BtoZ1] = coeff[k,l]
            DATA_TwoBody_Qubits[i*MATRIX_ZZ.shape[0]+k+Z1toZ2,i*MATRIX_ZZ.shape[1]+l+Z1toZ2] = coeff[k,l]


# 一体相互作用の組み込みを行う
for i in range(DIM):
    coeff = DATA_OneBody[i]*BECTOR_B
    for j in range(BECTOR_B.shape[0]):
        DATA_OneBody_Qubits[i*BECTOR_B.shape[0]+j] = coeff[j]

for i in range(DIM):
    coeff = LAMBDA*BECTOR_Z
    for j in range(BECTOR_Z.shape[0]):
        DATA_OneBody_Qubits[i*BECTOR_Z.shape[0]+j+BtoZ1] = coeff[j]
        DATA_OneBody_Qubits[i*BECTOR_Z.shape[0]+j+Z1toZ2] = coeff[j]



                               
""" 
量子モンテカルロシミュレーション
"""

if __name__ == '__main__':
    config_at_init_time = list




