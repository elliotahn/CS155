import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Set2helper import *
from sklearn.linear_model import LogisticRegression, Ridge, Lasso

def LogVsRidge():
    X, y = load_data('data/problem1data1.txt')
    m1 = LogisticRegression()
    m2 = Ridge()

    m1.fit(X, y)
    m2.fit(X, y)

    plot_data(X, y, m1, 'logregress.png', 'Logistic Regression')
    plot_data(X, y, m2, 'ridregress.png', 'Ridge Regression')

def LogVsHinge():
    X = np.array([[1, 0.5, 3], [1, 2, -2], [1, -3, 1]])
    y = np.array([1, 1, -1])
    w = np.array([0, 1, 0])

    for i, x in enumerate(X):
        dlog = grad_logloss(x, y[i], w)
        dhinge = grad_hingeloss(x, y[i], w)
        print(F'({x[1]}, {x[2]}):\t{dlog}\t{dhinge}')

def LogisticRegularization():
    X1, y1 = load_data('data/wine_training1.txt', ylabel = '#y')
    X2, y2 = load_data('data/wine_training2.txt', ylabel = '#y')
    Xtest, ytest = load_data('data/wine_testing.txt', ylabel = '#y')
    X = (X1, X2)
    y = (y1, y2)
    lam0 = 0.00001
    lamblst = 5 ** np.arange(15) * lam0

    # Three variables which we want to plot against lambda
    Einlst = [[], []]
    Eoutlst = [[], []]
    wnormlst = [[], []]

    # Fit the weights for each lambda
    for lamb in lamblst:
        print(F'Lambda = {lamb}')
        m1 = RegLogReg()
        m2 = RegLogReg()
        m = (m1, m2)

        for i in range(2):
            m[i].fit(X[i], y[i], lam = lamb)
            Einlst[i].append(m[i].get_error(X[i], y[i]))
            Eoutlst[i].append(m[i].get_error(Xtest, ytest))
            wnormlst[i].append(np.linalg.norm(m[i].w))

    i_min = np.argmin(Eoutlst[0])
    i_min2 = np.argmin(Eoutlst[1])
    print('Training 1')
    print(lamblst[i_min], Einlst[0][i_min], Eoutlst[0][i_min])
    print(Eoutlst[0])

    print('\nTraining 2')
    print(lamblst[i_min2], Einlst[1][i_min2], Eoutlst[1][i_min2])
    print(Eoutlst[1])

    xlim = [lamblst[0], lamblst[-1]]
    basic_line_plot(lamblst, Einlst, fname = 'regEin.png',
        xlim = xlim, xlabel = r'$\lambda$', ylabel = 'Error', 
        kind = 'logx', legend = ['Training1', 'Training2'])
    basic_line_plot(lamblst, Eoutlst, fname = 'regEout.png',
        xlim = xlim, xlabel = r'$\lambda$', ylabel = 'Error', 
        kind = 'logx', legend = ['Training1', 'Training2'])
    basic_line_plot(lamblst, wnormlst, fname = 'regwnorm.png',
        xlim = xlim, xlabel = r'$\lambda$', ylabel = '||w||', 
        kind = 'logx', legend = ['Training1', 'Training2'])

def LassoVsRidge():
    X, y = load_data('data/problem3data.txt', sep = '\t')
    alphalst = np.linspace(0.1, 3, 30)

    alphaplot(X, y, Lasso, alphalst, 'lassoalpha.png',
                leg_loc = 'upper right')

    alphalst = np.arange(1, 1e4 + 1)
    alphaplot(X, y, Ridge, alphalst, 'ridgealpha.png',
                leg_loc = 'upper right')

def alphaplot(X, y, mtype, alphalst, fname, leg_loc = 'best'):
    An = len(alphalst)
    w = np.zeros(X.shape[1] * An).reshape(An, X.shape[1])
    for i, alpha in enumerate(alphalst):
        m = mtype(alpha)
        m.fit(X, y)
        w[i] = m.coef_

    sns.set_style('whitegrid')
    fig = plt.figure()
    ax = plt.gca()
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('w')
    ax.set_xlim(alphalst[0], alphalst[-1])

    for i in range(w.shape[1]):
        ax.plot(alphalst, w[:, i])
    plt.legend([F'w{i}' for i in range(1, An + 1)], loc = leg_loc)
    plt.savefig(fname, bbox_inches = 'tight', dpi = 300)
    plt.close()

def testpython():
    size = 10
    x = np.random.random(size)
    x = (x - x.mean()) / x.std()
    xarr = x.reshape(size, 1)
    w = np.random.random() * 4 - 2
    lamb = 100
    y = w * x + np.random.normal(loc = 0, scale = lamb, size = size)
    print(F'xT y : {x.dot(y)}')

    print(F'True w : {w}')
    myw = mypred(x, y, lamb)
    print(F'My w : {myw}')
    m = Lasso(fit_intercept = False, max_iter = 2000)
    m.fit(xarr, y)
    wpred = m.coef_[0]
    print(F'Predicted w : {wpred}')

    tloss = myloss(x, y, w, lamb)
    mloss = myloss(x, y, myw, lamb)
    sloss = myloss(x, y, wpred, lamb)
    print(F'True Loss : {tloss}')
    print(F'My Loss : {mloss}')
    print(F'Sklearn Loss : {sloss}')

    if mloss == min([tloss, mloss, sloss]):
        return True
    else:
        return False



def myloss(x, y, w, lamb):
    return np.linalg.norm(y - x.dot(w))**2 + lamb * np.abs(w)

def mypred(x, y, lamb):
    xty = x.dot(y)
    xx = x.dot(x)
    if 2 * xty > lamb:
        return (2 * xty - lamb) / (2 * xx)
    elif 2 * xty + lamb < 0:
        return (2 * xty + lamb) / (2 * xx)
    else:
        return 0

if __name__=='__main__':
    #LogVsHinge()
    #LogVsRidge(X, y)
    #LogisticRegularization()

    #LassoVsRidge()
    iterations = 25
    me = 0
    for i in range(iterations):
        if(testpython()):
            me += 1
        print('\n\n')

    print(F'{me} / {iterations}')
