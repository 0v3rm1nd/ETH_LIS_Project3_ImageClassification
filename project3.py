__author__ = 'Mind'
import h5py
import numpy as np
import pandas as pd
from sklearn import preprocessing
import sklearn.metrics as skmet
import sklearn.grid_search as skgs
from sklearn.decomposition import PCA, RandomizedPCA
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve

# plot learning curves
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# visualize principle component/variance ratio
def pca_variance_ratio(X):
    """
    input:
    X: training/validation matrix to be normalized
    output:
    Plot with the number of components and the cumulative explained variance
    """
    pca = PCA().fit(X)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()

    return X


# feature scale
def feature_scale(X):
    """
    input:
    X: training/validation matrix to be normalized
    output:
    X: normalized matrix (no normalization is applied to the categorical data)
    """
    X = preprocessing.scale(X)
    return X


# score function
def classification_score(Y_pred, Y_true):
    """
    input:
    Ypred: matrix with predicted values
    Ytrue: matrix with actual values
    output:
    accuracy score
    """
    score = skmet.accuracy_score(Y_true, Y_pred)
    return score


# SVM gaussian kernel + cross validation
def gaussian_svm(X_tr_pca, Y, X_test_pca):
    """
    input:
    X_train: feature scaled and normalized matrix - training set
    Y_train: training labels matrix
    X_test: feature scaled and normalized matrix - validation set
    output:
    Y_pred: matrix with predictions over the validation set
    """
    gaussian_svm = svm.SVC(C=10)
    scorefun = skmet.make_scorer(classification_score)
    param_grid = {}
    grid_search = skgs.GridSearchCV(gaussian_svm, param_grid, scoring=scorefun, cv=5)
    grid_search.fit(X_tr_pca, Y)
    score_Y = grid_search.best_score_
    best_Y = grid_search.best_estimator_
    # predict based on the best estimator for Y1
    Y_pred = best_Y.predict(X_test_pca)
    # overall cross validation score
    cv_score = 1 - score_Y
    print('C-V score =', cv_score)

    return Y_pred

# get the shape of the datasets
file_train = h5py.File("train.h5", "r")
file_valid = h5py.File("test.h5", "r")

# get data as np arrays
X = file_train["data"].value
Y = file_train["label"].value
# convert labels to a single array
Y = Y.ravel()
X_valid = file_valid["data"].value
file_train.close()
file_valid.close()

# pca/variance ratio --> get the number of principle components to use
# pca_variance_ratio(X)

# plot learning curve
# title = 'Learning Curves (LogReg)'
# estimator = linear_model.LogisticRegression(penalty='l1', dual=False)
# cv = cross_validation.ShuffleSplit(X_tr_pca.shape[0], n_iter=5,
#                                    test_size=0.2, random_state=0)
# plot_learning_curve(estimator, title, X_tr_pca, Y, cv=cv)
# plt.show()

# feature scale
X = feature_scale(X)
X_valid = feature_scale(X_valid)

pca = RandomizedPCA(n_components=500, whiten=True).fit(X)
X_tr_pca = pca.transform(X)
X_test_pca = pca.transform(X_valid)

# gaussian svm + cross validation
Y_pred = gaussian_svm(X_tr_pca, Y, X_test_pca)

# output to csv
Y_pred = pd.DataFrame(data=Y_pred, dtype=int)
Y_pred.to_csv('test_out.csv', index=False, header=False)
