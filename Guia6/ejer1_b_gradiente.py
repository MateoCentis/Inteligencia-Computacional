import numpy as np
import matplotlib.pyplot as plt

def F(x,y):
    z = np.power(x**2+y**2,0.25)*(np.power(np.sin(50*np.power((x**2+y**2),0.1)),2)+1)
    return z

def gradient_descent(gradient, start_x, start_y, learn_rate, n_iter):
    vector_x = start_x
    vector_y = start_y
    for _ in range(n_iter):
        diff_x = -learn_rate * gradient(vector_x, vector_y)
        vector_x += diff_x

        diff_y = -learn_rate * gradient(vector_y, vector_x)
        vector_y += diff_y

        if (_ + 1) % 10 == 0:
            print(min(F(vector_x, vector_y)))

    return vector_x, vector_y

def gradient(x, y):
    term1 = (x * (np.sin(50 * (x**2 + y**2)**0.1)**2 + 1)) / (2 * (x**2 + y**2)**0.75)
    term2 = (20 * x * np.cos(50 * (x**2 + y**2)**0.1) * np.sin(50 * (x**2 + y**2)**0.1)) / (x**2 + y**2)**(13/20)
    return term1 + term2

start_x = np.random.uniform(-100, 100, 100)
start_y = np.random.uniform(-100, 100, 100)
learn_rate = 0.000005
n_iter = 100

vector_x, vector_y = gradient_descent(gradient, start_x, start_y, learn_rate, n_iter)
index_min = np.argmin(F(vector_x, vector_y))
print(start_x[index_min], start_y[index_min], F(start_x[index_min], start_y[index_min]))