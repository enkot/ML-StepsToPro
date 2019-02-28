#%% [markdown]
# ## Visualize the multivariate linear regression
#%% [markdown]
# Testing implemented multivariate linear regression algorithm. Here you can see how prediction line fit the data using Gradient descent.

#%%
#Get and normalize data

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation
from linear_regression import LinearRegressor
from utils import min_max_scaler

X1, X2, y = np.loadtxt('linear_regression/data1.txt', delimiter=',', unpack=True)
m = len(y)
X = min_max_scaler(X1) # we use first feature for this example

regressor = LinearRegressor()
regressor.fit(X, y)
regressor.hypothesisHistory

fig, ax = plt.subplots()
ax.scatter(X, y)
line, = ax.plot(X, regressor.hypothesisHistory[298], color='red')

# If you want to save Gradient Descent steps as animation 
def animate(i):
    line.set_ydata(X)
    line.set_ydata(regressor.hypothesisHistory[i])
    ax.set_title(f'Learning iteration {i}')
    return line,

anim = animation.FuncAnimation(
    fig, animate, frames=len(regressor.hypothesisHistory) - 1, interval=10, blit=False)
anim.save('animation.gif', writer='imagemagick', fps=30)

plt.show()


#%%



