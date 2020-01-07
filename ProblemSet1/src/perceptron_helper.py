########################################
# CS/CNS/EE 155 2018
# Problem Set 1
#
# Author:       Joey Hong
# Description:  Set 1 Perceptron helper
########################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('white')

class perceptronTracker:
    def __init__(self, X, Y):
        d = X.shape[1]
        self.X = X
        self.Y = Y
        out_ind = [''] + ['W'] * d + ['X'] * d + ['']
        in_ind = ['b'] + ['w{}'.format(i + 1) for i in range(d)] + \
                ['x{}'.format(i + 1) for i in range(d)] + ['y']
        self.hier_index = pd.MultiIndex.from_tuples(list(zip(out_ind, in_ind)))
        self.df = pd.DataFrame(index = self.hier_index).transpose()
        self.df.index.names = ['t']

    def df_add(self, row):
        rowadd = np.array(row)
        dfadd = pd.DataFrame(rowadd.T, index = self.hier_index).transpose()

        self.df = self.df.append(dfadd, ignore_index = True)
        self.df.index.names = ['t']

    def returndf(self):
        return self.df

    def returnw(self):
        t = self.df.shape[0]
        return np.array(self.df['W'][t])

    def returnX(self):
        t = self.df.shape[0]
        return np.array(self.df['X'][t])

    def returnY(self):
        t = self.df.shape[0]
        return self.df['', 'y'][t]

def predict(x, w, b):
    '''
    The method takes the weight vector and bias of a perceptron model, and
    predicts the label for a single point x.
    
    Inputs:
        x: A (D, ) shaped numpy array containing a single point.
        w: A (D, ) shaped numpy array containing the weight vector.
        b: A float containing the bias term.
    
    Output:
       The label (1 or -1) for the point x.
    '''
    prod = np.dot(w, x) + b
    return 1 if prod >= 0 else -1


def plot_data(X, Y, ax):
    # This method plots a labeled (with -1 or 1) 2D dataset.
    ax.scatter(X[Y == 1, 0], X[Y == 1, 1], c = 'green', marker='+')
    ax.scatter(X[Y == -1, 0], X[Y == -1, 1], c = 'red')


def boundary(x_1, w, b):
    # Gets the corresponding x_2 value given x_1 and the decision boundary of a
    # perceptron model. Note this only works for a 2D perceptron.
    if w[1] == 0.0:
        denom = 1e-6
    else:
        denom = w[1]

    return (-w[0] * x_1 - b) / denom

def plot_perceptron_data(i, ax, m):
    X = m.X
    Y = m.Y
    ax.clear()

    maxlim = [0, 0]
    minlim = [0, 0]
    bound_scale = 0.05
    for j in [0, 1]:
        minlim[j] = min(X[:, j])
        maxlim[j] = max(X[:, j])

        minlim[j] -= (maxlim[j] - minlim[j]) * bound_scale
        maxlim[j] += (maxlim[j] - minlim[j]) * bound_scale

    ax.scatter(X[Y == 1, 0], X[Y == 1, 1], c = 'blue', marker = '+')
    ax.scatter(X[Y == -1, 0], X[Y == -1, 1], c = 'red', s = 4)

    X_mis = np.array(m.df['X'].loc[i])
    if not np.isnan(X_mis).any():
        ax.scatter(X_mis[0], X_mis[1], s = 60, facecolors = 'none',
                    edgecolors = 'purple')

    w = m.df['W'].loc[i]
    b = m.df['', 'b'].loc[i]

    x = np.linspace(minlim[0], maxlim[0], 1000)
    if w[1] == 0:
        denom = 1e-6
    else:
        denom = w[1]

    y = -(w[0] * x + b) / denom

    ax.plot(x, y, color = 'k', linewidth = 1)
    if np.sign(np.dot([minlim[0], minlim[1]], w) + b) == -1:
        ax.fill_between(x, minlim[1], y, facecolor = 'red', alpha = 0.3)
    else:
        ax.fill_between(x, y, maxlim[1], facecolor = 'red', alpha = 0.3)
    ax.set_xlim(minlim[0], maxlim[0])
    ax.set_ylim(minlim[1], maxlim[1])
    ax.set_title('Perceptron Algorithm')

def plot_perceptron(w, b, ax):
    # This method plots a perceptron decision boundary line. Note this only works for 
    # 2D perceptron.    
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    
    x_2s = [boundary(x_1, w, b) for x_1 in xlim]
    ax.plot(xlim, x_2s)
    if predict([xlim[0], ylim[0]], w, b) == -1:
        ax.fill_between(xlim, ylim[0], x_2s, facecolor='red', alpha=0.5)
    else:
        ax.fill_between(xlim, x_2s, ylim[-1], facecolor='red', alpha=0.5)
