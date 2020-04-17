import numpy as np
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy as sp
from numpy import median
 
# 3 практикум
# 2 вариант

'''
 #1
sygm = 20 # параметр нормального распределения
Theta1 = lambda x: median(x)
Theta2 = lambda x: np.mean(x)
Theta3 = lambda x, sygm: np.mean(x) / (1 + (sygm * len(x))**(-1))
#квадратичная функция потерь
funcloss = lambda x, theta: (x - theta) ** 2
# квадратичный риск
sqrisk1 = []
sqrisk2 = []
sqrisk3 = []
# абсолютный риск
absrisk1 = []
absrisk2 = []
absrisk3 = []
plt.figure(figsize=(13, 13))
for theta in np.arange(1, 1.1, 0.001):
    # вектор квадратичной функции потерь
    sqfunc1 = []
    sqfunc2 = []
    sqfunc3 = []
    # вектор абсолютной функции потерь
    absfunc1 = []
    absfunc2 = []
    absfunc3 = []
    i = 0
    while (i <= 100):
        t = np.random.uniform(0, theta, 50) # генерируем выборку
        # строим оценки
        th1 = Theta1(t)
        th2 = Theta2(t)
        th3 = Theta3(t, sygm)
        # задаем вектор квадратичной функции потерь
        sqfunc1.append(funcloss(th1, theta))
        sqfunc2.append(funcloss(th2, theta))
        sqfunc3.append(funcloss(th3, theta))
        #задаем вектор абсолютной функции потерь
        absfunc1.append(math.fabs(th1 - theta))
        absfunc2.append(math.fabs(th2 - theta))
        absfunc3.append(math.fabs(th3 - theta))
        i += 1

    # чтобы получить риск усредняем по всем значениям вектора функции потерь
    sqrisk1.append(np.mean(sqfunc1))
    sqrisk2.append(np.mean(sqfunc2))
    sqrisk3.append(np.mean(sqfunc3))

    absrisk1.append(np.mean(absfunc1))
    absrisk2.append(np.mean(absfunc2))
    absrisk3.append(np.mean(absfunc3))

plt.subplot(2, 1, 1)
x_axis = np.arange(1, 1.1, 0.001)
# для теты1 - желтый график
# для теты2 - красный график
# для теты3 - синий график
plt.plot(x_axis, sqrisk1, 'y')
plt.plot(x_axis, sqrisk2, 'r')
plt.plot(x_axis, sqrisk3, 'b')
plt.subplot(2, 1, 2)
plt.plot(x_axis, absrisk1, 'y')
plt.plot(x_axis, absrisk2, 'r')
plt.plot(x_axis, absrisk3, 'b')
plt.show()



#2

def normal(mu, sygm, x):
    return 1/(np.sqrt(2 * np.pi * sygm)) * np.exp( - (x - mu)**2 / (2 * sygm))

theta = 3
plt.figure(figsize = (15, 15))
N = [5, 10, 25, 50]
sygm = [0.5, 1, 5, 10]
num = 0
for s in sygm:
    for n in N:
        x = np.random.normal(theta, 1, n)
        x_axis = np.linspace(1, 5, n)
        num += 1
        g = plt.subplot(4, 4, num)
        g.set_ylabel(s)
        g.set_xlabel(n)
        prior = normal(0, s, x_axis)
        # парематеры апостериорной плотности из таблицы:
        m = 1/(1/s + n/s)*(np.sum(x)/s)
        sy = (1/s + n/s)**(-1)
        posterior = normal(m, sy, x_axis)
        plt.plot(x_axis, prior, 'r')
        plt.plot(x_axis, posterior, 'b')
plt.show()
'''
#3
N = [5, 10, 25, 50]
sygm = [0.5, 1, 5, 10]
theta = 4
for s in sygm:
    check = 0
    for n in N:
        k = 0
        while (k < 1000):
            x = np.random.uniform(0, theta, n)
            th1 = Theta1(x)
            th2 = Theta2(x)
            if (math.fabs(theta - th1) > math.fabs(theta - th2)):
                check += 1
            elif (math.fabs(theta - th1) < math.fabs(theta - th2)):
                check -= 1
            k += 1
        if (check > 0):
            print('s = %f, n = %d, th1' % (s, n))
        elif (check < 0):
            print('s = %f, n = %d, th2' % (s, n))
        else:
            print('s = %f, n = %d, eq' % (s, n))


'''
#6
absrisk = []
sqrisk = []
N = [5, 10, 25, 50]
sygm = [0.5, 1, 5, 10]
theta = 2
from scipy import stats
for s in sygm:
    for n in N:
        xx = np.random.normal(theta, 1, n)
        # апостериорные параметры:
        m = 1 / (1 / s + n / s) * (np.sum(xx) / s)
        sy = (1 / s + n / s) ** (-1)
        # байесовская оценка для квадратичного риска это матожидание
        # то есть апостериорный параметр
        theta1 = m
        # байесовская оценка для абсолютного риска это
        # значение функции обратной к функции распределения в точке 1/2

        theta2 = stats.norm.ppf(0.5, m, sy)

        absrisk.append(theta2)
        sqrisk.append(theta1)


sb.distplot(absrisk)
#plt.show()
sb.distplot(sqrisk)
plt.show()
'''