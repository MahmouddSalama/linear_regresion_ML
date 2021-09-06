import numpy as np
import matplotlib.pyplot as plt
def model( a,b,x):
    return a*x+b

def loss_function(a,b,x,y):
    num=len(x)
    predection=model(a,b,x)
    return (.5/num)*(np.square(predection-y)).sum()
Lr=1e-4
def optimal(a,b,x,y):
    num=len(x)
    predection=model(a,b,x)
    da=(1.0/num)*((predection-y)*x).sum()
    db=(1.0/num)*(predection-y).sum()
    a=a-Lr*da
    b=b-Lr*db
    return a,b

def  iteritor(a,b,x,y,itme):
    for i in range(itme):
        a,b=optimal(a,b,x,y)

    return a,b

a=np.random.rand(1)
b=np.random.rand(1)
x=[3,21,22,34,54,34,55,67,89,99]
y=[1,10,14,34,44,36,22,67,79,90]
x=np.array(x)
y=np.array(y)

a,b=iteritor(a,b,x,y,100)
predection=model(a,b,x)
loss=loss_function(a,b,x,y)
plt.scatter(x,y)
plt.plot(x,predection)
plt.show()

