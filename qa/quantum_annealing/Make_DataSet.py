# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from scipy.stats import zscore

#--------------------------------------------------------------------------------
# lassoで利用する

LAMBDA = 1
M_PENALTY = 10

class lasso:
    def lasso_con(self, Lam=LAMBDA, Pena=M_PENALTY):
        self.twobody_con = np.zeros((3*self.DIM, 3*self.DIM))
        self.onebody_con = np.zeros(3*self.DIM)

        for i in range(self.DIM): # 最小二乗法の部分
            self.twobody_con[i,i] = sum(np.square(self.DATA_X[:,i]))
            for j in range(i):
                self.twobody_con[j,i] = 2*sum(self.DATA_X[:,i]*self.DATA_X[:,j])
            self.onebody_con[i] = -2*sum(self.DATA_X[:,i]*self.DATA_Y[:])

            self.twobody_con/(2*self.NumData)
            self.onebody_con/(2*self.NumData)

        for i in range(self.DIM): # l1-normの部分
            self.twobody_con[i,i] += Lam*Pena
            self.twobody_con[self.DIM+i,self.DIM+i] = Lam*Pena
            self.twobody_con[2*self.DIM+i,2*self.DIM+i] = Lam*Pena
            self.twobody_con[i,self.DIM+i] = 2*Lam*Pena
            self.twobody_con[i,2*self.DIM+i] = -2*Lam*Pena
            self.twobody_con[self.DIM+i,2*self.DIM+i] = -2*Lam*Pena
            
            self.onebody_con[self.DIM+i] = Lam
            self.onebody_con[2*self.DIM+i] = Lam

    def lasso_dis(self, array1, array2):
        self.array_beta = array1
        self.array_z = array2
        self.twobody_dis = np.zeros((self.DIM*(len(array1)+2*len(array2)), self.DIM*(len(array1)+2*len(array2))))
        self.onebody_dis = np.zeros(self.DIM*(len(array1)+2*len(array2)))
        
        Matrix_11 = np.dot(array1.reshape(-1,1), array1.reshape(1,-1))
        Matrix_12 = np.dot(array1.reshape(-1,1), array2.reshape(1,-1))
        Matrix_22 = np.dot(array2.reshape(-1,1), array2.reshape(1,-1))

        for i in range(self.DIM):
            for j in range(i+1):
                self.twobody_dis[len(array1)*j:len(array1)*(j+1), len(array1)*i:len(array1)*(i+1)] = self.twobody_con[j,i]*Matrix_11

            self.twobody_dis[len(array1)*i:len(array1)*(i+1), self.DIM*len(array1)+len(array2)*i:self.DIM*len(array1)+len(array2)*(i+1)] = self.twobody_con[i,self.DIM+i]*Matrix_12
            self.twobody_dis[len(array1)*i:len(array1)*(i+1), self.DIM*(len(array1)+len(array2))+len(array2)*i:self.DIM*(len(array1)+len(array2))+len(array2)*(i+1)] = self.twobody_con[i,2*self.DIM+i]*Matrix_12

            self.twobody_dis[self.DIM*len(array1)+len(array2)*i:self.DIM*len(array1)+len(array2)*(i+1), self.DIM*len(array1)+len(array2)*i:self.DIM*len(array1)+len(array2)*(i+1)] = self.twobody_con[self.DIM+i,self.DIM+i]*Matrix_22
            self.twobody_dis[self.DIM*len(array1)+len(array2)*i:self.DIM*len(array1)+len(array2)*(i+1), self.DIM*(len(array1)+len(array2))+len(array2)*i:self.DIM*(len(array1)+len(array2))+len(array2)*(i+1)] = self.twobody_con[self.DIM+i,2*self.DIM+i]*Matrix_22
            self.twobody_dis[self.DIM*(len(array1)+len(array2))+len(array2)*i:self.DIM*(len(array1)+len(array2))+len(array2)*(i+1), self.DIM*(len(array1)+len(array2))+len(array2)*i:self.DIM*(len(array1)+len(array2))+len(array2)*(i+1)] = self.twobody_con[2*self.DIM+i,2*self.DIM+i]*Matrix_22

            self.onebody_dis[len(array1)*i:len(array1)*(i+1)] = self.onebody_con[i]*array1
            self.onebody_dis[self.DIM*len(array1)+len(array2)*i:self.DIM*len(array1)+len(array2)*(i+1)] = self.onebody_con[self.DIM+i]*array2
            self.onebody_dis[self.DIM*(len(array1)+len(array2))+len(array2)*i:self.DIM*(len(array1)+len(array2))+len(array2)*(i+1)] = self.onebody_con[2*self.DIM+i]*array2

    def Print_spin(self, spin):
        beta = np.zeros(self.DIM)
        z1 = np.zeros(self.DIM)
        z2 = np.zeros(self.DIM)

        for i in range(self.DIM):
            beta[i] = np.dot(spin[len(self.array_beta)*i:len(self.array_beta)*(i+1)],self.array_beta)
            z1[i] = np.dot(spin[self.DIM*len(self.array_beta)+len(self.array_z)*i:self.DIM*len(self.array_beta)+len(self.array_z)*(i+1)],self.array_z)
            z2[i] = np.dot(spin[self.DIM*(len(self.array_beta)+len(self.array_z))+len(self.array_z)*i:self.DIM*(len(self.array_beta)+len(self.array_z))+len(self.array_z)*(i+1)],self.array_z)

        print(beta)
        print(z1)
        print(z2)

        
#------------------------------------------------------------------------------

class make_dataset(lasso):
    def __init__(self, file_name):
        self.FILE_DATA = pd.read_csv(file_name, index_col=0)
        self.FILE_COLUMNS_X_NAME = list(self.FILE_DATA.columns[0:-1])
        self.FILE_COLUMNS_Y_NAME = [self.FILE_DATA.columns[-1]]
        self.DATA_X = zscore(self.FILE_DATA.iloc[:,:-1].values)
        self.DATA_Y = self.FILE_DATA.iloc[:,-1].values
        self.DIM = self.DATA_X.shape[1]
        self.NumData = self.DATA_X.shape[0]

        
