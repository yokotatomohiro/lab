#coding:utf-8
import time
import math
import numpy as np
import os
import random
import matplotlib.pyplot as plt

# パラメータの入力
TROTTER_DIM = int(input("Trotter dimension: "))
ANN_PARA =  float(input("initial annealing parameter: "))
ANN_STEP = int(input("Annealing Step: "))
MC_STEP = int(input("MC step: "))
BETA = float(input("inverse Temperature: "))
REDUC_PARA = 0.99

"""
都市についてのデータ(都市の番号とx,y座標)を取得
"""
FILE_NAME = 'dj38.tsp'

f = open(os.path.dirname(os.path.abspath(FILE_NAME))+'/'+FILE_NAME).read().split("\n")

POINT = []
for i in f:
    POINT.append(i.split(" "))

# 都市データ
NCITY = len(POINT)
TOTAL_TIME = NCITY
print(NCITY)
for i in range(NCITY):
    POINT[i].remove(POINT[i][0])
print(POINT)
for i in range(NCITY):
    for j in range(2):
        POINT[i][j] = float(POINT[i][j])

## 2都市間の距離
def distance(point1, point2):
    return math.sqrt((point1[1]-point2[1])**2 + (point1[0]-point2[0])**2)


## 全体のspinの配位
## spinの各次元数は[TROTTER_DIM, TOTAL_TIME, NCITY]で表す
def getSpinConfig():

    ## あるトロッタ次元の、ある時刻におけるspinの配位
    def spin_config_at_a_time_in_a_TROTTER_DIM(tag):
        config = list(-np.ones(NCITY, dtype = np.int))
        config[tag] = 1
        return config


    ## あるトロッタ次元におけるspinの配位
    def spin_config_in_a_TROTTER_DIM(tag):
        spin = []
        spin.append(config_at_init_time)
        for i in xrange(TOTAL_TIME-1):
            spin.append(list(spin_config_at_a_time_in_a_TROTTER_DIM(tag[i])))
        return spin

    spin = []
    for i in xrange(TROTTER_DIM):
        tag = np.arange(1,NCITY)
        np.random.shuffle(tag)
        spin.append(spin_config_in_a_TROTTER_DIM(tag)) 
    return spin


# 最短距離となるTrotter次元を選び、その時のルートを出力
def getBestRoute(config):   
    length = []
    for i in xrange(TROTTER_DIM):
        route = []
        for j in xrange(TOTAL_TIME):
            route.append(config[i][j].index(1))
        length.append(getTotaldistance(route))

    min_Tro_dim = np.argmin(length)
    Best_Route = []
    for i in xrange(TOTAL_TIME):
        Best_Route.append(config[min_Tro_dim][i].index(1))
    return Best_Route


## あるルートに対するmax値で割った総距離
def getTotaldistance(route):
    Total_distance = 0
    for i in xrange(TOTAL_TIME):
        Total_distance += distance(POINT[route[i]],POINT[route[(i+1)%TOTAL_TIME]])/max_distance
    return Total_distance


## あるルートに対する総距離
def getRealTotaldistance(route):
    Total_distance = 0
    for i in xrange(TOTAL_TIME):
        Total_distance += distance(POINT[route[i]], POINT[route[(i+1)%TOTAL_TIME]])
    return Total_distance


## 量子モンテカルロステップ
def QMC_move(config, ann_para):
    # 異なる2つの時刻a,bを選ぶ
    c = np.random.randint(0,TROTTER_DIM)
    a_ = range(1,TOTAL_TIME)
    a = np.random.choice(a_)
    a_.remove(a)
    b = np.random.choice(a_)

    # あるトロッタ数cで、時刻a,bにいる都市p,q
    p = config[c][a].index(1)
    q = config[c][b].index(1)

    # エネルギー差の値を初期化
    delta_cost = delta_costc = delta_costq_1 = delta_costq_2 = delta_costq_3 = delta_costq_4 = 0

    # (7)式の第一項についてspinをフリップする前後のエネルギーの差
    for j in range(NCITY):
        l_p_j = distance(POINT[p], POINT[j])/max_distance
        l_q_j = distance(POINT[q], POINT[j])/max_distance
        delta_costc += 2*(-l_p_j*config[c][a][p] - l_q_j*config[c][a][q])*(config[c][a-1][j]+config[c][(a+1)%TOTAL_TIME][j])+2*(-l_p_j*config[c][b][p] - l_q_j*config[c][b][q])*(config[c][b-1][j]+config[c][(b+1)%TOTAL_TIME][j])

    # (7)式の第二項についてspinをフリップする前後のエネルギー差
    para = (1/BETA)*math.log(math.cosh(BETA*ann_para/TROTTER_DIM)/math.sinh(BETA*ann_para/TROTTER_DIM))
    delta_costq_1 = config[c][a][p]*(config[(c-1)%TROTTER_DIM][a][p]+config[(c+1)%TROTTER_DIM][a][p])
    delta_costq_2 = config[c][a][q]*(config[(c-1)%TROTTER_DIM][a][q]+config[(c+1)%TROTTER_DIM][a][q])
    delta_costq_3 = config[c][b][p]*(config[(c-1)%TROTTER_DIM][b][p]+config[(c+1)%TROTTER_DIM][b][p])
    delta_costq_4 = config[c][b][q]*(config[(c-1)%TROTTER_DIM][b][q]+config[(c+1)%TROTTER_DIM][b][q])

    # (7)式についてspinをフリップする前後のエネルギー差
    delta_cost = delta_costc/TROTTER_DIM+para*(delta_costq_1+delta_costq_2+delta_costq_3+delta_costq_4)

    # 確率min(1,exp(-BETA*delta_cost))でflip
    if delta_cost <= 0:
        config[c][a][p]*=-1
        config[c][a][q]*=-1
        config[c][b][p]*=-1
        config[c][b][q]*=-1
    elif np.random.random() < np.exp(-BETA*delta_cost):
        config[c][a][p]*=-1
        config[c][a][q]*=-1
        config[c][b][p]*=-1
        config[c][b][q]*=-1

    return config


"""
量子アニーリングシミュレーション
"""
if __name__ == '__main__':

    # 2都市間の距離の最大値
    max_distance = 0
    for i in range(NCITY):
        for j in range(NCITY):
            if max_distance <= distance(POINT[i], POINT[j]):
                max_distance = distance(POINT[i], POINT[j])


    # 初期時刻におけるspinの配位(初期時刻では必ず都市0にいる)
    config_at_init_time = list(-np.ones(NCITY,dtype=np.int))
    config_at_init_time[0] = 1

    print("start...")
    t0 = time.clock()

    np.random.seed(100)
    spin = getSpinConfig()
    LengthList = []
    for t in range(ANN_STEP):
        for i in range(MC_STEP):
            con = QMC_move(spin, ANN_PARA)
            rou = getBestRoute(con)
            length = getRealTotaldistance(rou)
        LengthList.append(length)
        print("Step:{}, Annealing Parameter:{}, length:{}".format(t+1,ANN_PARA, length))
        ANN_PARA *= REDUC_PARA

    Route = getBestRoute(spin)
    Total_Length = getRealTotaldistance(Route)
    elapsed_time = time.clock()-t0

    print("最短ルートは:{}".format(Route))
    print("最短距離は{}".format(Total_Length))
    print("処理時間は{}s".format(elapsed_time))

    plt.plot(LengthList)
    plt.show()
