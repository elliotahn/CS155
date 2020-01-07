import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, Image

from sgd_helper import (
    generate_dataset1,
    generate_dataset2,
    plot_dataset,
    plot_loss_function,
    animate_convergence,
    animate_sgd_suite
)

sns.set_style('dark')

def loss(X, Y, w):
    '''
    Calculate the squared loss function.
    
    Inputs:
        X: A (N, D) shaped numpy array containing the data points.
        Y: A (N, ) shaped numpy array containing the (float) labels of the data points.
        w: A (D, ) shaped numpy array containing the weight vector.
    
    Outputs:
        The loss evaluated with respect to X, Y, and w.
    '''
    
    return np.sqrt(np.linalg.norm(Y - np.matmul(X, w)))

def gradient(x, y, w):
    '''
    Calculate the gradient of the loss function with respect to the weight vector w,
    evaluated at a single point (x, y) and weight vector w.
    
    Inputs:
        x: A (D, ) shaped numpy array containing a single data point.
        y: The float label for the data point.
        w: A (D, ) shaped numpy array containing the weight vector.
        
    Output:
        The gradient of the loss with respect to w. 
    '''
    return -2 * (y - np.dot(w, x)) * x

def SGD(X, Y, w_start, eta, N_epochs, w_lst = True):
    '''
    Perform SGD using dataset (X, Y), initial weight vector w_start,
    learning rate eta, and N_epochs epochs.
    
    Inputs:
        X: A (N, D) shaped numpy array containing the data points.
        Y: A (N, ) shaped numpy array containing the (float) labels of the data points.
        w_start:  A (D, ) shaped numpy array containing the weight vector initialization.
        eta: The step size.
        N_epochs: The number of epochs (iterations) to run SGD.
        
    Outputs:
        W: A (N_epochs, D) shaped array containing the weight vectors from all iterations.
        losses: A (N_epochs, ) shaped array containing the losses from all iterations.
    '''
    
    N = X.shape[0]
    w = w_start.copy()
    wlst = []
    losses = []

    for iteration in range(N_epochs):
        index = np.random.permutation(np.arange(N))
        for i in index:
            w = w - eta * gradient(X[i], Y[i], w)
        wlst.append(w)
        losses.append(loss(X, Y, w))

    if w_lst:
        return np.array(wlst), np.array(losses)
    else:
        return wlst[-1], np.array(losses)

def animation1(X1, X2, Y1, Y2):
    # Parameters to feed the SGD.
    # <FR> changes the animation speed.
    params = ({'w_start': [0.01, 0.01], 'eta': 0.00001},)
    N_epochs = 1000
    FR = 20

    # Let's animate it!
    anim = animate_sgd_suite(SGD, loss, X1, Y1, params, N_epochs, FR)
    anim.save('animation1.gif', fps=30, writer = 'imagemagick')
    HTML('<img src="animation1.gif">')

def animation2(X1, X2, Y1, Y2):
    # Parameters to feed the SGD.
    params = ({'w_start': [0.01, 0.01], 'eta': 0.00001},)
    N_epochs = 1000
    FR = 20

    # Let's do it!
    W, _ = SGD(X1, Y1, params[0]['w_start'], params[0]['eta'], N_epochs)
    anim = animate_convergence(X1, Y1, W, FR)
    anim.save('animation2.gif', fps=30, writer='imagemagick')
    HTML('<img src="animation2.gif">')

def animation3(X1, X2, Y1, Y2):
    # Parameters to feed the SGD.
    # Here, we specify each different set of initializations as a dictionary.
    params = (
        {'w_start': [-0.8, -0.3], 'eta': 0.00001},
        {'w_start': [-0.9, 0.4], 'eta': 0.00001},
        {'w_start': [-0.4, 0.9], 'eta': 0.00001},
        {'w_start': [0.8, 0.8], 'eta': 0.00001},
    )
    N_epochs = 1000
    FR = 20

    # Let's go!
    anim = animate_sgd_suite(SGD, loss, X1, Y1, params, N_epochs, FR)
    anim.save('animation3.gif', fps=30, writer='imagemagick')
    HTML('<img src="animation3.gif">')

def animation4(X1, X2, Y1, Y2):
    # Parameters to feed the SGD.
    params = (
        {'w_start': [-0.8, -0.3], 'eta': 0.00001},
        {'w_start': [-0.9, 0.4], 'eta': 0.00001},
        {'w_start': [-0.4, 0.9], 'eta': 0.00001},
        {'w_start': [0.8, 0.8], 'eta': 0.00001},
    )
    N_epochs = 1000
    FR = 20

    # Animate!
    anim = animate_sgd_suite(SGD, loss, X2, Y2, params, N_epochs, FR)
    anim.save('animation4.gif', fps=30, writer='imagemagick')
    HTML('<img src="animation4.gif">')

def animation5(X1, X2, Y1, Y2):
    # Parameters to feed the SGD.
    params = (
        {'w_start': [0.7, 0.8], 'eta': 0.00001},
        { 'w_start': [0.2, 0.8], 'eta': 0.00005},
        {'w_start': [-0.2, 0.7], 'eta': 0.0001},
        {'w_start': [-0.6, 0.6], 'eta': 0.0002},
    )
    N_epochs = 100
    FR = 2

    # Go!
    anim = animate_sgd_suite(SGD, loss, X1, Y1, params, N_epochs, FR, ms=2)
    anim.save('animation5.gif', fps=30, writer='imagemagick')
    HTML('<img src="animation5.gif">')

def epochlearning(X1, X2, Y1, Y2):
    '''Plotting SGD convergence'''

    #==============================================
    # TODO: For the given learning rates, plot the 
    # loss for each epoch.
    #==============================================

    eta_vals = [1e-6, 5e-6, 1e-5, 3e-5, 1e-4]
    w_start = [0.01, 0.01]
    N_epochs = 1000

    losslst = []
    for eta in eta_vals:
        losslst.append(SGD(X1, Y1, w_start, eta, N_epochs)[1])

    sns.set_style('whitegrid')

    fig = plt.figure()
    ax = plt.gca()

    for i in range(len(eta_vals)):
        ax.plot(range(1, N_epochs + 1), losslst[i])

    ax.set_title('Learning Rate for Different $\eta$')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Training Error')
    ax.legend(eta_vals)
    plt.savefig('epochlearning.png', bbox_inches = 'tight')

    plt.close()

def animation6(X1, X2, Y1, Y2):
    # Parameters to feed the SGD.
    params = ({'w_start': [0.01, 0.01], 'eta': 1},)
    N_epochs = 100
    FR = 2

    # Voila!
    anim = animate_sgd_suite(SGD, loss, X1, Y1, params, N_epochs, FR, ms=2)
    anim.save('animation6.gif', fps=30, writer='imagemagick')
    HTML('<img src="animation6.gif">')

def bigeta(X1, X2, Y1, Y2):
    # Parameters to feed the SGD.
    w_start = [0.01, 0.01]
    eta = 10
    N_epochs = 100

    # Presto!
    W, losses = SGD(X1, Y1, w_start, eta, N_epochs)

def animation7(X1, X2, Y1, Y2):
    # Import different SGD & loss functions.
    # In particular, the loss function has multiple optima.
    from sgd_multiopt_helper import SGD, loss

    # Parameters to feed the SGD.
    params = (
        {'w_start': [0.9, 0.9], 'eta': 0.01},
        { 'w_start': [0.0, 0.0], 'eta': 0.01},
        {'w_start': [-0.8, 0.6], 'eta': 0.01},
        {'w_start': [-0.8, -0.6], 'eta': 0.01},
        {'w_start': [-0.4, -0.3], 'eta': 0.01},
    )
    N_epochs = 100
    FR = 2

    # One more time!
    anim = animate_sgd_suite(SGD, loss, X1, Y1, params, N_epochs, FR, ms=2)
    anim.save('animation7.gif', fps=30, writer='imagemagick')
    HTML('<img src="animation7.gif">')

def load_data(filename, bias = False, N = None):
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
        if bias:
            X = np.append(np.ones(1), X)
    else:
        if bias:
            nr = X.shape[0]
            X = np.hstack((np.ones(nr).reshape(nr, 1), X))
    
    return X, y

def new_data_set():
    X, y = load_data('data/sgd_data.csv', bias = True)
    w_start = np.array([0.001, 0.001, 0.001, 0.001, 0.001])
    N_epochs = 1000
    
    eta_lst = np.exp(np.array([-8, -10, -11, -12, -13, -14, -15]))
    eta = np.exp(-15)
    w, loss = SGD(X, y, w_start, eta, N_epochs, w_lst = False)
    print(w)
    w_an = analytical_sol(X, y)
    print(w_an)
    print(np.linalg.norm(w - w_an) / np.linalg.norm(w_an))
    #new_epoch_learning(X, y, w_start, eta_lst, N_epochs)

def analytical_sol(X, y):
    return np.ravel(np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y)))

def new_epoch_learning(X, y, w_start, eta_lst, N_epochs):
    sns.set_style('whitegrid')
    fig = plt.figure()
    ax = plt.gca()

    epoch_lst = np.arange(1, N_epochs + 1)
    for eta in eta_lst:
        w, loss = SGD(X, y, w_start, eta, N_epochs, w_lst = False)
        ax.plot(epoch_lst, loss)

    ax.set_xlim(0, 250)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Training Error')
    ax.set_title('Learning Rate For sgd_data')
    ax.legend([np.format_float_scientific(n, precision = 2) for n in eta_lst])
    plt.savefig('epochlearning2.png', bbox_inches = 'tight', dpi = 300)
    plt.close()


if __name__ == '__main__':
    X1, Y1 = generate_dataset1()
    X2, Y2 = generate_dataset2()

    #animation1(X1, X2, Y1, Y2)
    #animation2(X1, X2, Y1, Y2)
    #animation3(X1, X2, Y1, Y2)
    #animation4(X1, X2, Y1, Y2)
    #animation5(X1, X2, Y1, Y2)
    #epochlearning(X1, X2, Y1, Y2)
    #animation6(X1, X2, Y1, Y2)
    #bigeta(X1, X2, Y1, Y2)
    #animation7(X1, X2, Y1, Y2)
    new_data_set()





    