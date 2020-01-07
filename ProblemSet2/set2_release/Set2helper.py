import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import SGDClassifier

class RegLogReg:
    def __init__(self, fit_intercept = True):
        self.w = None
        self.fit_intercept = fit_intercept

    def standardize(self, X):
        N, d = X.shape
        Z = np.zeros(N * d).reshape(N, d)
        for i in range(d):
            Xc = X[:, i]
            Z[:, i] = (Xc - Xc.mean()) / Xc.std()

        return Z

    def add_bias(self, X):
        row = X.shape[0]
        return np.hstack((np.ones(row).reshape(row, 1), X))

    def fit(self, X, y, lam = 0):
        Xt = self.standardize(X)
        if self.fit_intercept:
            Xt = self.add_bias(Xt)
        N_epochs = 20000
        eta = 5e-4
        index_array = np.arange(Xt.shape[0])
        lamb = lam / Xt.shape[0]

        w = np.random.random(Xt.shape[1]) - 0.5

        for iteration in range(N_epochs):
            ind = np.random.permutation(index_array)
            for i in ind:
                w = w - eta * grad_logloss(Xt[i], y[i], w, lam = lamb)

        self.w = w

    def predict(self, X):
        ypred = []
        Xt = self.standardize(X)
        if self.fit_intercept:
            Xt = self.add_bias(Xt)
        for x in Xt:
            p = 1 / (1 + np.exp(-self.w.dot(x)))
            if p < 0.5:
                ypred.append(-1)
            else:
                ypred.append(1)

        return np.array(ypred)

    def get_error(self, X, y, lam = 0):
        Xt = self.standardize(X)
        if self.fit_intercept:
            Xt = self.add_bias(Xt)
        return logloss(Xt, y, self.w, lam)

def load_data(filename, sep = ',', ylabel = None):
    if ylabel is None:
        df = pd.read_csv(filename, sep = sep, header = None)
    else:
        df = pd.read_csv(filename, sep = sep)
    row, col = df.shape
    if col == 1:
        print('File only has one column')
        return
    if ylabel is None:
        y = df[df.columns[-1]]
        X = df[df.columns[:col - 1]]
    else:
        y = df[ylabel]
        X = df.drop(ylabel, axis = 1)
    y = np.array(y)
    if col == 2:
        X = np.ravel(X)
    else:
        X = np.array(X)
    return X, y

def plot_data(X, y, m, filename, title = None):
    sns.set_style('white')
    fig = plt.figure()
    ax = plt.gca()

    scale = 0.05
    xmin = min(X[:, 0])
    xmax = max(X[:, 0])
    ymin = min(X[:, 1])
    ymax = max(X[:, 1])
    dx = xmax - xmin
    dy = ymax - ymin

    xmin = xmin - scale * dx
    xmax = xmax + scale * dx
    ymin = ymin - scale * dy
    ymax = ymax + scale * dy

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    if title is not None:
        ax.set_title(title)

    ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color = 'red', 
                                    edgecolor = 'k', s = 15)
    ax.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color = 'blue',
                                    edgecolor = 'k', s = 15)

    ax.legend([1, -1])

    density = 1000
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, density),
                            np.linspace(ymin, ymax, density))

    Z = m.predict(np.c_[np.ravel(xx), np.ravel(yy)]).reshape(xx.shape)
    ax.contour(xx, yy, Z, 0, linewidths = 1, colors = 'k')

    plt.savefig(filename, bbox_inches = 'tight', dpi = 300)
    plt.close()

def grad_logloss(x, y, w, lam = 0):
    return 2 * lam * w - y * x / (1 + np.exp(y * w.dot(x)))

def logloss(X, y, w, lam = 0):
    loss = lam * w.dot(w)
    for i, x in enumerate(X):
        loss += np.log(1 + np.exp(-y[i] * w.dot(x)))

    return loss

def grad_hingeloss(x, y, w, lam = 0):
    if 1 - y * w.dot(x) > 0:
        dloss = -y * x
    else:
        dloss = 0
    return dloss + lam * w

def bias(X):
    row, col = X.shape
    return np.hstack((np.ones(row).reshape(row, 1), X))

def basic_line_plot(x, ylst, fname = 'needtitle.png', xlim = None, 
                    ylim = None, title = None, xlabel = None, ylabel = None, 
                    kind = None, legend = None):
    sns.set_style('whitegrid')
    fig = plt.figure()
    ax = plt.gca()

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if kind is None:
        for y in ylst:
            ax.plot(x, y)
    elif kind == 'logx':
        for y in ylst:
            ax.semilogx(x, y)
        ax.set_xscale('log')
    elif kind == 'logy':
        for y in ylst:
            ax.semilogy(x, y)
        ax.set_yscale('log')
    elif kind == 'loglog':
        ax.set_xscale('log')
        ax.set_yscale('log')
        for y in ylst:
            ax.loglog(x, y)

    if legend is not None:
        ax.legend(legend)

    plt.savefig(fname, bbox_inches = 'tight', dpi = 300)
    plt.close()

