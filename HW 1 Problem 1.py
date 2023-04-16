
import numpy as np # imports a fast numerical programming library
import scipy as sp #imports stats functions, amongst other things
import matplotlib as mpl # this actually imports matplotlib
import matplotlib.cm as cm #allows us easy access to colormaps
import matplotlib.pyplot as plt #sets up plotting under plt
import pandas as pd #lets us handle data as dataframes
from statistics import variance as variance
import warnings

#sets up pandas table display
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)

import seaborn as sns #sets up styles and gives us more plotting options

#suppress warnings
warnings.filterwarnings('ignore')
df_x = pd.read_csv(r"C:\Users\Richard\Documents\ML HW 1\logistic_x.txt", sep="\ +", names=["x1","x2"], header=None, engine='python')
df_y = pd.read_csv(r'C:\Users\Richard\Documents\ML HW 1\logistic_y.txt', sep='\ +', names=["y"], header=None, engine='python')
df_y = df_y.astype(int)
df_x.head()
log_terms = []

x = np.hstack([np.ones((df_x.shape[0], 1)), df_x[["x1","x2"]].values])
y = df_y["y"].values
m = len(y)
theta = np.zeros(x.shape[1])


# Sigmoid, Gradient, and Hessian functions:
def sigmoid(x):
    # returns a single value
    return 1/(1 + np.exp(-x))

def gradient(theta, x, y):
    z = y*x.dot(theta)
    g = -np.mean((1-sigmoid(z))*y*x.T, axis=1)
    return g

def hessian(theta, x, y):
    # returns a 3x3 matrix
    dim1 = x.shape[1]
    dim2 = x.shape[0]
    z = y*x.dot(theta)
    hess = np.zeros((x.shape[1], x.shape[1]))

    for i in range(hess.shape[0]): 
        for j in range(hess.shape[0]): 
            if i <= j:
                hess[i,j] = np.mean(sigmoid(z)*(1-sigmoid(z)) * x[:,i] * x[:,j])
            if i != j:
                    hess[j][i] = hess[i][j] 
    return hess

def newton(theta, x, y, eps):
    delta = 1
    while delta > eps:
        theta_prev = theta.copy()
        # print(hessian(theta,x,y))
        theta -= np.linalg.inv(hessian(theta, x, y)).dot(gradient(theta, x, y))
        delta = np.linalg.norm(theta - theta_prev, ord = 1)
    return theta

theta_final = newton(theta, x, y, 1e-6)

df_x.insert(0, "y", df_y)
df_x["y"] = pd.to_numeric(df_x["y"],downcast='signed')
df_x.head()

# Generate vector to plot decision boundary
x1_vec = np.linspace(df_x["x1"].min(),df_x["x1"].max(),2)

# Plot raw data
sns.scatterplot(x="x1", y="x2", hue="y", data=df_x)

# Plot decision boundary
plt.plot(x1_vec,(-x1_vec*theta_final[1]-theta_final[0])/theta_final[2], color="red")

plt.show()