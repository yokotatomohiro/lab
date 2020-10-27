# -*- coding:utf-8 -*-

import sys
import numpy as np
import csv

from Make_DataSet import lasso, make_dataset
from Quantum_Annealing import qmc

case1 = make_dataset('boston.csv')
case1.lasso_con()
array1 = np.array([-128,64,32,16,8,4,2,1,0.5,0.25])
array2 = np.array([4,2,1,0.5,0.25])
case1.lasso_dis(array1, array2)
monte1 = qmc(case1.twobody_dis, case1.onebody_dis)

case2 = make_dataset('boston.csv')
case2.lasso_con()
array1 = np.array([-4,-2,-1,-0.5,-0.25,0.25,0.5,1,2,4])
array2 = np.array([4,2,1,0.5,0.25])
case2.lasso_dis(array1, array2)
monte2 = qmc(case2.twobody_dis, case2.onebody_dis)

case3 = make_dataset('boston.csv')
case3.lasso_con()
array1 = np.array([-1,-1,-0.5,-0.5,-0.25,0.25,0.5,0.5,1,1])
array2 = np.array([1,1,0.5,0.5,0.25])
case3.lasso_dis(array1, array2)
monte3 = qmc(case3.twobody_dis, case3.onebody_dis)

case4 = make_dataset('boston.csv')
case4.lasso_con()
array1 = np.array([-1,-0.5,-0.5,-0.25,-0.25,0.25,0.25,0.5,0.5,1])
array2 = np.array([1,0.5,0.5,0.25,0.25])
case4.lasso_dis(array1, array2)
monte4 = qmc(case4.twobody_dis, case4.onebody_dis)

case5 = make_dataset('boston.csv')
case5.lasso_con()
array1 = np.array([-4,2,1,0.5,0.25])
array2 = np.array([2,1,0.5,0.25])
case5.lasso_dis(array1, array2)
monte5 = qmc(case5.twobody_dis, case5.onebody_dis)


print('trotter=2')
monte1.Quantum_Annealing(2)
case1.Print_spin(monte1.result_spin)

monte2.Quantum_Annealing(2)
case2.Print_spin(monte2.result_spin)

monte3.Quantum_Annealing(2)
case3.Print_spin(monte3.result_spin)

monte4.Quantum_Annealing(2)
case4.Print_spin(monte4.result_spin)

monte5.Quantum_Annealing(2)
case5.Print_spin(monte5.result_spin)

f = open('trotter-2.csv', 'w')
dataWriter = csv.writer(f)
for i in range(len(monte1.cost_point)):
    dataWriter.writerow([monte1.cost_point[i],monte2.cost_point[i],monte3.cost_point[i],monte4.cost_point[i],monte5.cost_point[i]])
f.close()

print('trotter=4')
monte1.Quantum_Annealing(4)
case1.Print_spin(monte1.result_spin)

monte2.Quantum_Annealing(4)
case2.Print_spin(monte2.result_spin)

monte3.Quantum_Annealing(4)
case3.Print_spin(monte3.result_spin)

monte4.Quantum_Annealing(4)
case4.Print_spin(monte4.result_spin)

monte5.Quantum_Annealing(4)
case5.Print_spin(monte5.result_spin)

f = open('trotter-4.csv', 'w')
dataWriter = csv.writer(f)
for i in range(len(monte1.cost_point)):
    dataWriter.writerow([monte1.cost_point[i],monte2.cost_point[i],monte3.cost_point[i],monte4.cost_point[i],monte5.cost_point[i]])
f.close()

print('trotter=8')
monte1.Quantum_Annealing(8)
case1.Print_spin(monte1.result_spin)

monte2.Quantum_Annealing(8)
case2.Print_spin(monte2.result_spin)

monte3.Quantum_Annealing(8)
case3.Print_spin(monte3.result_spin)

monte4.Quantum_Annealing(8)
case4.Print_spin(monte4.result_spin)

monte5.Quantum_Annealing(8)
case5.Print_spin(monte5.result_spin)

f = open('trotter-8.csv', 'w')
dataWriter = csv.writer(f)
for i in range(len(monte1.cost_point)):
    dataWriter.writerow([monte1.cost_point[i],monte2.cost_point[i],monte3.cost_point[i],monte4.cost_point[i],monte5.cost_point[i]])
f.close()






