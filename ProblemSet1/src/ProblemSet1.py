# CS155 PROBLEM SET 1

# IMPORT LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from perceptron_helper import *

sns.set_style('white')

def load_data(filename, N = None):
    '''
    Function gets data stored in a file and returns the inputs and target
    labels for that data set
    Input:
        filename: name of the file holding the data
        N (optional): Number of data points to be returned from the file
                        If N is not specified, the entire file is used
    Output:
        N input points X and corresponding labels y
    '''

    data = pd.read_csv(filename)
    size = data.shape[0]

    # If N is specified, randomize the rows and select N points
    if N is not None:
        if N < size:
            data = data.sample(frac = N / size)

    # Get the X and y labels
    X = np.array(data.drop('y', axis = 1))
    y = np.array(data['y'])

    # If X is one-dimensional, transform it to an array
    if X.shape[1] == 1:
        X = np.ravel(X)
    
    return X, y

def MSError(y1, y2):
    return np.sum((y1 - y2)**2) / len(y1)

def Validation_Error(d, X, y, Nfold = 5):
    '''
    Function returns cross-validation and training error from data X and y 
    given the degree of a polynomial for a polynomial model.
    Input:
        d: degree of polynomial model
        X: input of data
        y: target labels of data
        Nfold: Partitions for cross-validation
    Output:
        Ecv: cross-validation error
        Ein: training error
    '''
    # Run through cross-validation partitions
    kf = KFold(n_splits = Nfold)
    Eout = 0.
    Ein = 0.
    for train_index, test_index in kf.split(X):
        # Train the data
        coef = np.polyfit(X[train_index], y[train_index], d)
        # Get the validation error
        y_pred = np.polyval(coef, X[test_index])
        Eout += MSError(y[test_index], y_pred)
        # Get the training error
        y_pred = np.polyval(coef, X[train_index])
        Ein += MSError(y[train_index], y_pred)

    Eout /= Nfold
    Ein /= Nfold

    return Eout, Ein

def polyLearnCurve(d):
    '''
    Function generates a learning curve for polynomial model of degree d.
    The learning curve is for data size 20, 25, 30, 35,...,95, 100
    Input:
        d: degree of polynomial model
    Output:
        None - png figure is created showing the plot of the learning curve
    '''

    # List of N between 20 and 100 by 5
    Nlst = np.array(list(range(20, 101, 5)))
    Ecv = []
    Ein = []

    iterations = 50
    for N in Nlst:
        # Run iterations times and average out to get smoother results
        ECV = 0
        EIN = 0
        for i in range(iterations):
            # Get the data
            X, y = load_data('data/bv_data.csv', N)
            E1, E2 = Validation_Error(d, X, y)
            ECV += E1
            EIN += E2

        # Append the errors to list
        ECV /= iterations
        EIN /= iterations
        Ecv.append(ECV)
        Ein.append(EIN)

    # Convert lists to numpy arrays
    Ecv = np.array(Ecv)
    Ein = np.array(Ein)

    # Plot learning curve
    fig, ax = plt.subplots()
    ax.plot(Nlst, Ecv)
    ax.plot(Nlst, Ein)
    ax.set_xlabel('N')
    ax.set_ylabel('Mean Squared Error')
    ax.set_xlim(Nlst[0], Nlst[-1])
    if d == 6:
        ax.set_ylim(0.5, 2.5)
    if d == 12:
        ax.set_ylim(0.1, 2.2)
    ax.legend(['Ecv', 'Ein'])
    ax.set_title('Polynomial Learning Curve d = {}'.format(d))
    ax.grid()
    plt.savefig('figures/learncurve{}'.format(d), bbox_inches = 'tight',\
                             dpi = 300)
    plt.close()

def update_perceptron(m, X, Y, w, b):
    """
    This method updates a perceptron model. Takes in the previous weights
    and returns weights after an update, which could be nothing.
    
    Inputs:
        X: A (N, D) shaped numpy array containing a single point.
        Y: A (N, ) shaped numpy array containing the labels for the points.
        w: A (D, ) shaped numpy array containing the weight vector.
        b: A float containing the bias term.
    
    Output:
        next_w: A (D, ) shaped numpy array containing the next weight vector
                after updating on a single misclassified point, if one exists.
        next_b: The next float bias term after updating on a single
                misclassified point, if one exists.
    """
    next_w, next_b = np.copy(w), np.copy(b)

    k = -1
    
    for i in range(X.shape[0]):
        if predict(X[i], next_w, next_b) != Y[i]:
            k = i

            next_w = Y[i] * X[i] + next_w
            next_b += Y[i]
            break

    d = X.shape[1]
    wlst = [w[j] for j in range(d)]
    if k == -1:
        lst = [np.nan] * (d + 1)
        m.df_add([b] + wlst + lst)
    else:
        lst = [X[k][j] for j in range(d)] + [Y[k]]
        m.df_add([b] + wlst + lst)

    return next_w, next_b

def run_perceptron(X, Y, w, b, max_iter = 1000):
    """
    This method runs the perceptron learning algorithm. Takes in initial weights
    and runs max_iter update iterations. Returns final weights and bias.
    
    Inputs:
        X: A (N, D) shaped numpy array containing a single point.
        Y: A (N, ) shaped numpy array containing the labels for the points.
        w: A (D, ) shaped numpy array containing the initial weight vector.
        b: A float containing the initial bias term.
        max_iter: An int for the maximum number of updates evaluated.
        
    Output:
        w: A (D, ) shaped numpy array containing the final weight vector.
        b: The final float bias term.
    """

    m = perceptronTracker(X, Y)
    for i in range(max_iter):
        w_new, b_new = update_perceptron(m, X, Y, w, b)
        if (np.array_equal(w, w_new)) and (b == b_new):
            break
        
        w = w_new
        b = b_new

    return m

def problem2():
    for d in [1, 2, 6, 12]:
        polyLearnCurve(d)

def problem3():
    # Problem 3A
    '''
    X = np.array([[ -3, -1], [0, 3], [1, -2]])
    Y = np.array([ -1, 1, 1])'''

    # Problem 3C
    X = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
    Y = np.array([1, 1, -1, -1])


    w = np.array([0, 1])
    b = 0

    m = run_perceptron(X, Y, w, b, 16)

    print(m.returndf())

    fig, ax = plt.subplots()

    anim = FuncAnimation(fig, plot_perceptron_data, frames = m.df.shape[0],
            fargs = [ax, m], interval = 1500, repeat = False)

    # Problem 3A
    #anim.save('simpleperceptron.html')

    # Problem 3C
    anim.save('perceptronns.html')


if __name__ == '__main__':
    #problem2()
    problem3()
