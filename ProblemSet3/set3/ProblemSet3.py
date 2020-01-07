import numpy as np
import pandas as pd
from matplotlib import animation
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from Set3Helper import *
from boosting_helper import *


def Problem2():
	X, y = read_data('data/messidor_features.arff', 
    					skiprows = list(range(24)))

	leaf_lst = list(range(1, 26))
	leafsize, leaf_err = TreeErrors(X, y, DecisionTreeClassifier, leaf_lst,
					'Set3Figures/minleafsize.png','min_samples_leaf')

	print(F'Optimal minimum leaf size : {leafsize}\tError : {leaf_err}')

	depth_lst = list(range(2, 21))
	depth, depth_err = TreeErrors(X, y, DecisionTreeClassifier, depth_lst,
					'Set3Figures/maxdepth.png', 'max_depth')
	print(F'Optimal maximum depth : {depth}\tError : {depth_err}')

	leafrf, leaferrrf = TreeErrors(X, y, RandomForestClassifier, leaf_lst,
					'Set3Figures/minleafsizerf.png', 'min_samples_leaf',
					**{'n_estimators' : 1000})
	print(F'Optimal minimum leaf size RF : {leafrf}\tError : {leaferrrf}')

	depthrf, depth_errrf = TreeErrors(X, y, RandomForestClassifier, depth_lst,
					'Set3Figures/maxdepthrf.png', 'max_depth',
					**{'n_estimators' : 1000})
	print(F'Optimal maximum depth RF : {depthrf}\tError : {depth_errrf}')

def Problem3():
	(X_train, Y_train), (X_test, Y_test) = \
										generate_dataset(2000, 500, 1.5, 4.0)
	model = gb_suite(GradientBoosting, 500, X_train, Y_train, X_test, Y_test,
					'Set3Figures/treeboost')
	anim = animate_gb(model, X_train, Y_train, 
			'Training dataset predictions per iteration')
	anim.save('Set3Figures/gbanimation.gif')

	model, D = ab_suite(AdaBoost, 500, X_train, Y_train, X_test, Y_test,
					'Set3Figures/treeadaboost')
	anim = animate_ab(model, X_train, Y_train, D, 
				'Training dataset predictions per iteration')
	anim.save('Set3Figures/abanimation.gif')

if __name__ == '__main__':
    #linVsTree()
	#Problem2()
	Problem3()