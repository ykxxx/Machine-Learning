"""
Description : Titanic
"""

## IMPORTANT: Use only the provided packages!

## SOME SYNTAX HERE.   
## I will use the "@" symbols to refer to some variables and functions. 
## For example, for the 3 lines of code below
## x = 2
## y = x * 2 
## f(y)
## I will use @x and @y to refer to variable x and y, and @f to refer to function f

import math
import csv
from util import *
from collections import Counter

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics


######################################################################
# classes
######################################################################

class Classifier(object) :

    ## THIS IS SOME GENERIC CLASS, YOU DON'T NEED TO DO ANYTHING HERE. 

    """
    Classifier interface.
    """

    def fit(self, X, y):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()


class MajorityVoteClassifier(Classifier) : ## INHERITS FROM THE @CLASSIFIER

    def __init__(self) :
        """
        A classifier that always predicts the majority class.

        Attributes
        --------------------
            prediction_ -- majority class
        """
        self.prediction_ = None

    def fit(self, X, y) :
        """
        Build a majority vote classifier from the training set (X, y).

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes

        Returns
        --------------------
            self -- an instance of self
        """
        majority_val = Counter(y).most_common(1)[0][0]
        self.prediction_ = majority_val
        return self

    def predict(self, X):
        """
        Predict class values.

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples

        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.prediction_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")

        # n,d = X.shape ## get number of sample and dimension
        y = [self.prediction_] * X.shape[0]
        return y


class RandomClassifier(Classifier) :

    def __init__(self) :
        """
        A classifier that predicts according to the distribution of the classes.

        Attributes
        --------------------
            probabilities_ -- an array specifying probability to survive vs. not 
        """
        self.probabilities_ = None ## should have length 2 once you call @fit

    def fit(self, X, y):
        """
        Build a random classifier from the training set (X, y).

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes

        Returns
        --------------------
            self -- an instance of self
        """

        ### ========== TODO : START ========== ###
        # part b: set self.probabilities_ according to the training set
        # in simpler wordings, find the probability of survival vs. not
        p_survived = Counter(y)[1] / X.shape[0]
        self.probabilities_ = [1-p_survived, p_survived]

        ### ========== TODO : END ========== ###

        return self

    def predict(self, X, seed=1234):
        """
        Predict class values.

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            seed -- integer, random seed

        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.probabilities_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        np.random.seed(seed)

        ### ========== TODO : START ========== ###
        # part b: predict the class for each test example
        # hint: use np.random.choice (check the arguments of np.random.choice) to randomly pick a value based on the given probability array @self.probabilities_

        y = np.random.choice(2, X.shape[0], p=self.probabilities_)

        ### ========== TODO : END ========== ###

        return y


######################################################################
# functions
######################################################################
def plot_histograms(X, y, Xnames, yname) :
    n,d = X.shape  # n = number of examples, d =  number of features
    fig = plt.figure(figsize=(20,15))
    nrow = 3; ncol = 3
    for i in range(d) :
        fig.add_subplot (3,3,i)
        data, bins, align, labels = plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname, show = False)
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xnames[i])
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')

    plt.savefig ('histograms.pdf')


def plot_histogram(X, y, Xname, yname, show = True) :
    """
    Plots histogram of values in X grouped by y.

    Parameters
    --------------------
        X     -- numpy array of shape (n,d), feature values
        y     -- numpy array of shape (n,), target classes
        Xname -- string, name of feature
        yname -- string, name of target
    """

    # set up data for plotting
    targets = sorted(set(y))
    data = []; labels = []
    for target in targets :
        features = [X[i] for i in range(len(y)) if y[i] == target]
        data.append(features)
        labels.append('%s = %s' % (yname, target))

    # set up histogram bins
    features = set(X)
    nfeatures = len(features)
    test_range = list(range(int(math.floor(min(features))), int(math.ceil(max(features)))+1))
    if nfeatures < 10 and sorted(features) == test_range:
        bins = test_range + [test_range[-1] + 1] # add last bin
        align = 'left'
    else :
        bins = 10
        align = 'mid'

    # plot
    if show == True:
        plt.figure()
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xname)
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
        plt.show()

    return data, bins, align, labels


def error(clf, X, y, ntrials=100, test_size=0.2):
    """
    Computes the classifier error over a random split of the data,
    averaged over ntrials runs.

    Parameters
    --------------------
        clf         -- classifier
        X           -- numpy array of shape (n,d), features values
        y           -- numpy array of shape (n,), target classes
        ntrials     -- integer, number of trials

    Returns
    --------------------
        train_error -- float, training error
        test_error  -- float, test error
    """

    ### ========== TODO : START ========== ###
    # compute cross-validation error over ntrials
    # hint: use @train_test_split to split the data into train/test set 
    # xtrain, xtest, ytrain, ytest = train_test_split (X,y, test_size = test_size, random_state = i)
    # now you can call the @clf.fit (xtrain, ytrain) and then do prediction

    train_error = 0 ## average error over all the @ntrials
    test_error = 0
    train_scores = []; test_scores = []; ## tracking the error for each of the @ntrials, these array should have length 100 once you're done. 
    for i in range(ntrials):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i)
        clf.fit(X_train, y_train)  # fit training data using the classifier
        y_pred_train = clf.predict(X_train)  # take the classifier and run it on the training data
        train_scores.append(1 - metrics.accuracy_score(y_train, y_pred_train, normalize=True))
        y_pred_test = clf.predict(X_test)
        test_scores.append(1 - metrics.accuracy_score(y_test, y_pred_test, normalize=True))

    train_error = np.average(train_scores)
    test_error = np.average(test_scores)

    ### ========== TODO : END ========== ###

    return train_error, test_error


def write_predictions(y_pred, filename, yname=None) :
    """Write out predictions to csv file."""
    out = open(filename, 'wb')
    f = csv.writer(out)
    if yname :
        f.writerow([yname])
    f.writerows(list(zip(y_pred)))
    out.close()


######################################################################
# main
######################################################################

def main():
    # load Titanic dataset
    titanic = load_data("titanic_train.csv", header=1, predict_col=0)
    X = titanic.X; Xnames = titanic.Xnames
    y = titanic.y; yname = titanic.yname
    n,d = X.shape  # n = number of examples, d =  number of features



    #========================================
    # part a: plot histograms of each feature
    # print('Plotting...')
    # for i in range(d) :
    #     plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname)


    #========================================
    # train Majority Vote classifier on data
    print('Classifying using Majority Vote...')
    clf = MajorityVoteClassifier() # create MajorityVote classifier, which includes all model parameters
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)



    ### ========== TODO : START ========== ###
    # part b: evaluate training error of Random classifier
    print('Classifying using Random...')
    clf = RandomClassifier()
    clf.fit(X, y)
    y_pred = clf.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error using random classifier: %.3f' % train_error)

    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part c: evaluate training error of Decision Tree classifier
    # use criterion of "entropy" for Information gain
    print('Classifying using Decision Tree...')
    clf = DecisionTreeClassifier(criterion='entropy')
    clf.fit(X, y)
    y_pred = clf.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error using decision tree classifier: %.3f' % train_error)

    # call the function @DecisionTreeClassifier

    ### ========== TODO : END ========== ###



    # note: uncomment out the following lines to output the Decision Tree graph
    """
    # save the classifier -- requires GraphViz and pydot
    import StringIO, pydot
    from sklearn import tree
    dot_data = StringIO.StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=Xnames)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("dtree.pdf")
    """



    ### ========== TODO : START ========== ###
    # part d: evaluate training error of k-Nearest Neighbors classifier
    # use k = 3, 5, 7 for n_neighbors
    print('Classifying using k-Nearest Neighbors...')
    # call the function @KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error for k=3: %.3f' % train_error)

    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error for k=5: %.3f' % train_error)

    clf = KNeighborsClassifier(n_neighbors=7)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error for k=7: %.3f' % train_error)


    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part e: use cross-validation to compute average training and test error of classifiers
    print('Investigating various classifiers...')
    # call your function @error
    clf = MajorityVoteClassifier()
    train_error, test_error = error(clf, X, y)
    print('\t-- MajorityVote: training error = %.3f, test error = %.3f' % (train_error, test_error))

    clf = RandomClassifier()  # create Random classifier, which includes all model parameters
    train_error, test_error = error(clf, X, y)
    print('\t-- Random:       training error = %.3f, test error = %.3f' % (train_error, test_error))

    clf = DecisionTreeClassifier(
        criterion='entropy')  # create DecisionTree classifier, which includes all model parameters
    train_error, test_error = error(clf, X, y)
    print('\t-- DecisionTree: training error = %.3f, test error = %.3f' % (train_error, test_error))

    clf = KNeighborsClassifier(n_neighbors=5)  # create KNN classifier, which includes all model parameters
    train_error, test_error = error(clf, X, y)
    print('\t-- KNeighbors:   training error = %.3f, test error = %.3f' % (train_error, test_error))

    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part f: use 10-fold cross-validation to find the best value of k for k-Nearest Neighbors classifier
    print('Finding the best k for KNeighbors classifier...')
    # hint: use the function @cross_val_score
    k = list(range(1,50,2))
    error_rate = [] ## track accuracy for each value of $k, should have length 25 once you're done
    for i in k:
        clf = KNeighborsClassifier(n_neighbors=i)
        error_rate.append(1-np.average(cross_val_score(clf, X, y, cv=10)))

    plt.title('KNN 10-fold cross validation error rate')
    plt.xlabel('k')
    plt.ylabel('cross validation error rate')
    plt.plot(k, error_rate, marker='o')
    # plt.show()
    plt.savefig('./knn_cross_validation.png')
    plt.close()
    k_idx = np.argmin(error_rate)
    print('Best value of k is %d' % k[k_idx])

    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part g: investigate decision tree classifier with various depths
    print('Investigating depths...')

    depth = np.arange(1, 21, 1)
    train_errors = []
    test_errors = []

    for i in range(1, 21):
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=i)
        train_error, test_error = error(clf, X, y)
        train_errors.append(train_error)
        test_errors.append(test_error)

    plt.title('Decision tree cross validation score')
    plt.xlabel('Depth')
    plt.ylabel('cross validation error rate')
    plt.plot(depth, test_errors, marker='o', label='Test Error')
    plt.plot(depth, train_errors, marker='x', label='Training Error')
    plt.legend(loc='lower left')
    # plt.show()
    plt.savefig('./decision_tree_cross_validation.png')
    plt.close()
    d_idx = np.argmin(test_errors)
    print('Best depth is %d' % depth[d_idx])


    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part h: investigate Decision Tree and k-Nearest Neighbors classifier with various training set sizes
    print('Investigating training set sizes...')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    train_fraction = np.arange(0.1, 1.1, 0.1)
    avg_train_errors_dct = []
    avg_test_errors_dct = []
    avg_train_errors_knn = []
    avg_test_errors_knn = []

    for i in range(1, 11):
        dct = DecisionTreeClassifier(criterion='entropy', max_depth=3)
        knn = KNeighborsClassifier(n_neighbors=7)

        train_errors_dct = []
        test_errors_dct = []
        train_errors_knn = []
        test_errors_knn = []

        for j in range(100):

            if i < 10:
                X_train_subset, X_test_subset, y_train_subset, y_test_subset = train_test_split(X_train, y_train, test_size=0.1 * i, random_state=j)
            else:
                X_train_subset, y_train_subset = X_train, y_train

            dct.fit(X_train_subset, y_train_subset)
            y_predict_train = dct.predict(X_train_subset)
            train_errors_dct.append(metrics.accuracy_score(y_train_subset, y_predict_train, normalize=True))
            y_predict_test = dct.predict(X_test)
            test_errors_dct.append(metrics.accuracy_score(y_test, y_predict_test, normalize=True))

            knn.fit(X_train_subset, y_train_subset)
            y_predict_train = knn.predict(X_train_subset)
            train_errors_knn.append(metrics.accuracy_score(y_train_subset, y_predict_train, normalize=True))
            y_predict_test = knn.predict(X_test)
            test_errors_knn.append(metrics.accuracy_score(y_test, y_predict_test, normalize=True))

        avg_train_errors_dct.append(1-np.average(train_errors_dct))
        avg_test_errors_dct.append(1-np.average(test_errors_dct))
        avg_train_errors_knn.append(1-np.average(train_errors_knn))
        avg_test_errors_knn.append(1-np.average(test_errors_knn))

    plt.title('KNN and Decision Tree Learning Curves')
    plt.xlabel('Proportion of training set used')
    plt.ylabel('Error rate')
    ax = plt.gca()
    ax.set_ylim([0, 1])
    plt.plot(train_fraction, avg_train_errors_knn, marker='o', label='KNN Training Error')
    plt.plot(train_fraction, avg_test_errors_knn, marker='o', label='KNN Test Error')

    plt.plot(train_fraction, avg_train_errors_dct, marker='x', label='DT Training Error')
    plt.plot(train_fraction, avg_test_errors_dct, marker='x', label='DT Test Error')
    plt.legend(loc='upper right')
    # plt.show()
    plt.savefig('./learning_curve.png')
    plt.close()


    ### ========== TODO : END ========== ###


    print('Done')


if __name__ == "__main__":
    main()
