# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.animation as animation 
import math
import numpy as np
import sys

Tau = 1
N = 5
NumInte = 7
NumDec  = 4 # 整数部のビット数がNumInte、小数部のビット数がNumDec、符号部が1bit
DIM = 5

H   = None # 全体のハミルトニアン(エネルギー関数) 
H_B = None # ターゲット関数 make_○○○で問題を設定する

class Quantum_MonteCarlo: # 量子モンテカルロアルゴリズムの計算を行う
    def Make_DataSet() # データセットの作成を行う
    def Make_Matrix()
    
    def __init__(self, NumVar, Type): # Type:問題がグラフの場合'G', 連続変数の場合'V'を入力する
        if (Type == G):   # 問題がグラフの場合
            self.Type   = 'Graph'
            self.dim    = DIM
        elif (Type == V): # 問題が連続変数の場合
            self.Type   = 'Variable'
            self.dim    = DIM*(NumInte+NumDec+1)            
        else:             # 問題の指定がない場合システムを終了する
            print "問題のタイプgraph'G'もしくはvariable'V'を指定してください"
            exit()
        self.matrix = np.zeros(self.dim)
        

    def Set_problem_Graph(self, ):

    def Set_problem_Variable(self, NumVar, NumInte, NumDec) # NumData:変数の数, NumInte:整数部のビット数, NumDec:小数部のビット数
    

# ターゲット関数の作成　ガウス分布に従う乱数を使用した最小値探索問題
def make_random():
    diagonal_part = np.random.normal(0, scale=N / 2, size=2**N)
    print(diagonal_part)
    H_B = np.diag(diagonal_part)
    return H_B

# ターゲット関数の作成　巡回セールスマン問題
def make_TSL():
    N = P**2 # 導入する変数の数
    # 辺の重み行列
    M = numpy.symmetric((P, P)) # 対称行列の作成
    # 重みの設定 

# ターゲット関数の作成　整数計画問題
#def make_IP():

    
def scheduleE(time): # ターゲット関数のハミルトニアンの係数
    return time / Tau

def scheduleG(time): # 横磁場のハミルトニアンの係数
    return (Tau - time) / Tau

def create_tfim(time, hamiltonian):
    # ターゲット関数を加える
    v = scheduleE(time)
    hamiltonian = v * H_B

    # 非対角成分に対して横磁場を加える
    g = -1 * scheduleG(time)
    for i in range(2**N):
        for n in range(N):
            j = i ^ (1 << n) # iと1をnビットだけ左シフトしたものの排他的論理和
            hamiltonian[i, j] = hamiltonian[i, j] + g

    return hamiltonian
    

def amp2prob(vec):
    p = [np.abs(z)**2 for z in vec]
    return np.array(p)

if __name__ == '__main__':
    from scipy.linalg import eigh

    np.random.seed(0) # seed値を固定

    H = np.zeros(2**N)
    step = 0.001 # 0.01だと最適解に収束しにくかった
    time_steps = [step * i for i in range(int(Tau / step) + 1)]
    
    # 最適化したい問題を作成する関数を選択する
    H_B = make_random() # 最小値探索問題
    # H_B = make_TLS()    # 巡回セールスマン問題
    # H_B = make_IP()     # 整数計画問題

    
    # 確率振幅の棒グラフを表示する
    fig2 = plt.figure()
    ims2 = []
    x2 = np.array([i for i in range(2**N)]) # 確率振幅の横軸

    # ハミルトニアンを変化させずに計算を行った場合
    evals_B_all, evecs_B_all = eigh(H_B)
    y1 = amp2prob(evecs_B_all[:, 0])
    print(y1)
 
    # ハミルトニアンを変化させる
    for i, t in enumerate(time_steps):
        H = create_tfim(t, H)
        # evals_all:固有値, evecs_all:固有ベクトル
        # 固有値がエネルギー, 固有ベクトルが確率振幅にそれぞれ対応する
        evals_all, evecs_all = eigh(H)

        # 固有ベクトル(確率分布)をプロットする
        y2 = amp2prob(evecs_all[:, 0]) # 確率振幅の絶対値の二乗を計算することで確率する
        im2 = plt.bar(x2, y2, color="#337eb8")
        ims2.append(im2)
        if (t==1):
            print(y2)

    #ani1 = animation.ArtistAnimation(fig1, ims1, interval=100)
    #plt.show()
    ani2 = animation.ArtistAnimation(fig2, ims2, interval=100)
    plt.show()
