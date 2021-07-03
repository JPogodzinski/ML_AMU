import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import ipywidgets as widgets
import csv

reader = csv.reader(open('fires_thefts.csv'), delimiter=',')

x = list()
y = list()
for xi, yi in reader:
    x.append(float(xi))
    y.append(float(yi))


def h(theta, x):
    return theta[0] + theta[1] * x

def J(h, theta, x, y):
    """Cost fun"""
    m = len(y)
    return 1.0 / (2 * m) * sum((h(theta, x[i]) - y[i])**2 for i in range(m))

def costfun(fun, x, y):
    return lambda theta: J(fun, theta, x, y)

def gradient_descent(h, cost_fun, theta, x, y, alpha, eps):
    current_cost = cost_fun(h, theta, x, y)
    log = [[current_cost, theta]]
    m = len(y)
    while True:
        new_theta = [
            theta[0] - alpha/float(m) * sum(h(theta, x[i]) - y[i]
                                            for i in range(m)),
            theta[1] - alpha/float(m) * sum((h(theta, x[i]) - y[i]) * x[i]
                                            for i in range(m))]
        theta = new_theta
        try:
            prev_cost = current_cost
            current_cost = cost_fun(h, theta, x, y)
        except OverflowError:
            break
        if abs(prev_cost - current_cost) <= eps:
            break
        log.append([current_cost, theta])
    return theta, log

best_theta, log = gradient_descent(h, J, [0.0, 0.0], x, y, alpha=0.01, eps=0.00001)
print(best_theta)

x1=50
x2=100
x3=200
predicted_y1=h(best_theta,x1)
predicted_y2=h(best_theta, x2)
predicted_y3=h(best_theta,x3)

print(predicted_y1)
print(predicted_y2)
print(predicted_y3)


def gradient_descent2(h, cost_fun, theta, x, y, alpha, eps):
    list=[]
    current_cost = cost_fun(h, theta, x, y)
    list.append(current_cost)
    log = [[current_cost, theta]]
    m = len(y)
    for i in range (0,200):
        new_theta = [
            theta[0] - alpha/float(m) * sum(h(theta, x[i]) - y[i]
                                            for i in range(m)),
            theta[1] - alpha/float(m) * sum((h(theta, x[i]) - y[i]) * x[i]
                                            for i in range(m))]
        theta = new_theta
        try:
            prev_cost = current_cost
            list.append(prev_cost)
            current_cost = cost_fun(h, theta, x, y)
        except OverflowError:
            break
        if abs(prev_cost - current_cost) <= eps:
            break
        log.append([current_cost, theta])
    return theta, log, list


def gradient_descent3(h, cost_fun, theta, x, y, alpha, eps):
    list=[]
    current_cost = cost_fun(h, theta, x, y)
    list.append(current_cost)
    log = [[current_cost, theta]]
    m = len(y)
    for i in range (0,80):
        new_theta = [
            theta[0] - alpha/float(m) * sum(h(theta, x[i]) - y[i]
                                            for i in range(m)),
            theta[1] - alpha/float(m) * sum((h(theta, x[i]) - y[i]) * x[i]
                                            for i in range(m))]
        theta = new_theta
        try:
            prev_cost = current_cost
            list.append(prev_cost)
            current_cost = cost_fun(h, theta, x, y)
        except OverflowError:
            break
        if abs(prev_cost - current_cost) <= eps:
            break
        log.append([current_cost, theta])
    return theta, log, list


fig=plt.figure(figsize=(10,8))
show_x=np.arange(0,201,1)

theta, log, list1=gradient_descent2(h, J, [0.0, 0.0], x, y, alpha=0.001, eps=0.00001)
theta, log, list2=gradient_descent2(h, J, [0.0, 0.0], x, y, alpha=0.01, eps=0.00001)
theta, log, list3=gradient_descent3(h, J, [0.0, 0.0], x, y, alpha=0.1, eps=0.00001)

y1=np.array(list1)
y2=np.array(list2)
y3=np.array(list3)

print(show_x)
print(y1)
print(y2)
print(y3)

show_x3=np.arange(0,81,1)


ax1 = fig.add_subplot(3,1,1)
ax1.set_xlabel('Algorithm steps')
ax1.set_ylabel('J(θ)')
ax1.set_title('Task 3.2')

ax1.plot(show_x, y1, color='red', lw=2, label='α=0.001')
ax1.plot(show_x, y2, color='green', lw=2,label='α=0.01')
ax1.plot(show_x3, y3, color='#002d69', lw=2, label='α=0.1')
leg=ax1.legend()
plt.yscale('log')
plt.show()