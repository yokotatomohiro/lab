# -*- coding:utf-8 -*-
import time
import math
import numpy as np
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
TROTTER_DIM = int(input("トロッター次元数: m = ")) 
ANN_PARA = float(input("初期パラメータ: Γ_init = "))
ANN_STEP = int(input("アニーリング数: ANN = "))
MC_STEP = int(input("モンテカルロステップ: MC = "))
BETA = float(input("初期逆温度: β = "))
REDUCE_PARA = 0.99 # 減衰率

# データセットの内容を取得する
FILE_NAME = ''

f = open(os.path.dirname(os.path.abspath(FILE_NAME))+'/'+FILE_NAME).read().split("\n")

# データセットのファイル
X = [] # 計画行列
Y = [] # 解のベクトル


