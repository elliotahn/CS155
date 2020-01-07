import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import perceptron
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from boosting_helper import * 


# GENERATE AND PLOT DATA 

def gen_data(size, w = None):
    # Generate linear data
    X = (np.random.random(2 * size) * 2 - 1).reshape(size, 2)
    if w is None:
        w1, w2, x0, y0 = np.random.random(4) * 2 - 1
        y = np.sign(X[:, 1] - (w2 / w1) * (X[:, 0] - x0) - y0)
    elif w == 'random':
        y = np.random.choice([-1, 1], size)
    else:
        y = np.sign(X[:, 1] + (w[1] / w[2]) * X[:, 0] + w[0] / w[2])
    return X, y

def plot_data(X, y, fname, mlst = None, legend_loc = 'best'):
    sns.set_style('white')
    fig = plt.figure()
    ax = plt.gca()

    ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color = 'r', s = 8,
                        edgecolor = 'k')
    ax.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color = 'b', s = 8,
                        edgecolor = 'k')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.legend(['1', '-1'], loc = legend_loc)

    if mlst is not None:
        d = 400
        xx, yy = np.meshgrid(np.linspace(-1, 1, d), np.linspace(-1, 1, d))
        Xt = np.c_[np.ravel(xx), np.ravel(yy)]
        for m in mlst:
            Z = m.predict(Xt).reshape(d, d)
            ax.contour(xx, yy, Z, [0], linewidths = 1)

    plt.savefig(fname, bbox_inches = 'tight', dpi = 300)
    plt.close()

# COUNT NODES

def num_int_nodes(m):
    return m.tree_.node_count - m.get_n_leaves()

def internal_nodes():
    size = 100
    n_nodes = []

    for i in range(1000):
        X, y = gen_data(size)

        m = DecisionTreeClassifier(criterion = 'entropy')
        m.fit(X, y)
        n_nodes.append(num_int_nodes(m))

    print('Average : ', np.mean(n_nodes))
    print('Max : ', max(n_nodes))
    print('Std : ', np.std(n_nodes))

def linVstree():
    X, y = gen_data(150, w = np.array([0, 1, 1]))
    m1 = Perceptron()
    m2 = DecisionTreeClassifier()
    m1.fit(X, y)
    m2.fit(X, y)
    plot_data1(X, y, 'Set3Figures/random_linear.png', [m1, m2], 
                legend_loc = 'lower left')

# READ DATA FROM A FILE

def read_data(fname, header = None, skiprows = None, target = None):
    df = pd.read_csv(fname, header = header, skiprows = skiprows)
    
    if target is None:
        target = df.columns[-1]

    X = np.array(df.drop(target, axis = 1))
    y = np.array(df[target])

    if X.shape[1] == 1:
        X = np.ravel(X)

    return X, y

# DECISION TREE ERRORS

def TreeErrors(X, y, model, arglst, fname, treearg, **otherarg):
    Xtrain = X[:900]
    Xtest = X[900:]
    ytrain = y[:900]
    ytest = y[900:]

    Einlst = []
    Eoutlst = []

    for arg in arglst:
        m = model(criterion = 'gini', **{treearg : arg}, **otherarg)
        m.fit(Xtrain, ytrain)
        yinpred = m.predict(Xtrain)
        youtpred = m.predict(Xtest)

        Einlst.append(1 - accuracy_score(ytrain, yinpred))
        Eoutlst.append(1 - accuracy_score(ytest, youtpred))


    sns.set_style('whitegrid')
    fig = plt.figure()
    ax = plt.gca()

    ax.plot(arglst, Einlst)
    ax.plot(arglst, Eoutlst)
    ax.set_xlim(arglst[0], arglst[-1])
    ax.set_xlabel(treearg)
    ax.set_ylabel('Error')
    ax.legend(['Ein', 'Eout'])
    plt.savefig(fname, bbox_inches = 'tight', dpi = 300)
    plt.close()

    ind = np.argmin(Eoutlst)
    return arglst[ind], Eoutlst[ind]

# GRADIENT BOOSTING REGRESSION FOR CLASSIFICATION

class GradientBoosting():
    def __init__(self, n_clfs=100):
        '''
        Initialize the gradient boosting model.

        Inputs:
            n_clfs (default 100): Initializer for self.n_clfs.        
                
        Attributes:
            self.n_clfs: The number of DT weak regressors.
            self.clfs: A list of the DT weak regressors, initialized as empty.
        '''
        self.n_clfs = n_clfs
        self.clfs = []
        
    def fit(self, X, Y, n_nodes=4):
        '''
        Fit the gradient boosting model. Note that since we are implementing 
        this method in a class, rather than having a bunch of inputs and 
        outputs, you will deal with the attributes of the class.
        (see the __init__() method).
        
        This method should thus train self.n_clfs DT weak regressors and 
        store them in self.clfs.

        Inputs:
            X: A (N, D) shaped numpy array containing the data points.
            Y: A (N, ) shaped numpy array containing the (float) labels of 
                        the data points.
               (Even though the labels are ints, we treat them as floats.)
            n_nodes: The max number of nodes that the DT weak regressors are 
                        allowed to have.
        '''
        F = np.zeros(len(Y))

        for i in range(self.n_clfs):
            g = Y - F
            m = DecisionTreeRegressor(max_leaf_nodes = n_nodes)
            m.fit(X, g)
            self.clfs.append(m)
            Ym = m.predict(X)
            F = F + Ym


    def predict(self, X):
        '''
        Predict on the given dataset.

        Inputs:
            X: A (N, D) shaped numpy array containing the data points.

        Outputs:
            A (N, ) shaped numpy array containing the (float) labels of the 
            data points.
            (Even though the labels are ints, we treat them as floats.)
        '''
        # Initialize predictions.
        Y_pred = np.zeros(len(X))
        
        # Add predictions from each DT weak regressor.
        for clf in self.clfs:
            Y_curr = clf.predict(X)
            Y_pred += Y_curr

        # Return the sign of the predictions.
        return np.sign(Y_pred)

    def loss(self, X, Y):
        '''
        Calculate the classification loss.

        Inputs:
            X: A (N, D) shaped numpy array containing the data points.
            Y: A (N, ) shaped numpy array containing the (float) labels of 
                        the data points.
               (Even though the labels are ints, we treat them as floats.)
            
        Outputs:
            The classification loss.
        '''
        # Calculate the points where the predictions and the ground truths 
        # don't match.
        Y_pred = self.predict(X)
        misclassified = np.where(Y_pred != Y)[0]

        # Return the fraction of such points.
        return float(len(misclassified)) / len(X)

# ADABOOST FOR CLASSIFICATION

class AdaBoost():
    def __init__(self, n_clfs=100):
        '''
        Initialize the AdaBoost model.

        Inputs:
            n_clfs (default 100): Initializer for self.n_clfs.        
                
        Attributes:
            self.n_clfs: The number of DT weak classifiers.
            self.coefs: A list of the AdaBoost coefficients.
            self.clfs: A list of the DT weak classifiers, initialized as 
                        empty.
        '''
        self.n_clfs = n_clfs
        self.coefs = []
        self.clfs = []

    def fit(self, X, Y, n_nodes=4):
        '''
        Fit the AdaBoost model. Note that since we are implementing this 
        method in a class, rather than having a bunch of inputs and outputs, 
        you will deal with the attributes of the class.
        (see the __init__() method).
        
        This method should thus train self.n_clfs DT weak classifiers and 
        store them in self.clfs, with their coefficients in self.coefs.

        Inputs:
            X: A (N, D) shaped numpy array containing the data points.
            Y: A (N, ) shaped numpy array containing the (float) labels of 
                the data points.
               (Even though the labels are ints, we treat them as floats.)
            n_nodes: The max number of nodes that the DT weak classifiers are 
            allowed to have.
            
        Outputs:
            A (N, T) shaped numpy array, where T is the number of 
            iterations / DT weak classifiers, such that the t^th column 
            contains D_{t+1} (the dataset weights at iteration t+1).
        '''
        D = np.zeros(X.shape[0]) + 1 / X.shape[0]
        Dmat = np.zeros((X.shape[0], self.n_clfs))

        for i in range(self.n_clfs):
            m = DecisionTreeClassifier(max_leaf_nodes = n_nodes)
            m.fit(X, Y, sample_weight = D)
            ep = 1 - m.score(X, Y, sample_weight = D)
            alpha = 0.5 * np.log((1 - ep) / ep)
            ypred = m.predict(X)
            for j in range(X.shape[0]):
                D[j] *= np.exp(- alpha * Y[j] * ypred[j])

            self.clfs.append(m)
            self.coefs.append(alpha)
            D = D / np.linalg.norm(D, 1)
            Dmat[:, i] = D

        return Dmat


    
    def predict(self, X):
        '''
        Predict on the given dataset.

        Inputs:
            X: A (N, D) shaped numpy array containing the data points.
            
        Outputs:
            A (N, ) shaped numpy array containing the (float) labels of the 
            data points.
            (Even though the labels are ints, we treat them as floats.)
        '''
        # Initialize predictions.
        Y_pred = np.zeros(len(X))
        
        # Add predictions from each DT weak classifier.
        for i, clf in enumerate(self.clfs):
            Y_curr = self.coefs[i] * clf.predict(X)
            Y_pred += Y_curr

        # Return the sign of the predictions.
        return np.sign(Y_pred)

    def loss(self, X, Y):
        '''
        Calculate the classification loss.

        Inputs:
            X: A (N, D) shaped numpy array containing the data points.
            Y: A (N, ) shaped numpy array containing the (float) labels of 
                the data points.
               (Even though the labels are ints, we treat them as floats.)
            
        Outputs:
            The classification loss.
        '''
        # Calculate the points where the predictions and the ground truths 
        # don't match.
        Y_pred = self.predict(X)
        misclassified = np.where(Y_pred != Y)[0]

        # Return the fraction of such points.
        return float(len(misclassified)) / len(X)