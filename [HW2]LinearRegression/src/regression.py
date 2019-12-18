# This code was adapted from course material by Jenna Wiens (UMichigan).

# python libraries
import os

# numpy libraries
import numpy as np

# matplotlib libraries
import matplotlib.pyplot as plt
import time

######################################################################
# classes
######################################################################


class Data:
    def __init__(self, X=None, y=None):
        """
        Data class.
        
        Attributes
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
        """

        # n = number of examples, d = dimensionality
        self.X = X
        self.y = y

    def load(self, filename):
        """
        Load csv file into X array of features and y array of labels.
        
        Parameters
        --------------------
            filename -- string, filename
        """

        # determine filename
        dir = os.path.dirname("__file__")
        f = os.path.join(dir, "..", "data", filename)

        # load data
        with open(f, "r") as fid:
            data = np.loadtxt(fid, delimiter=",")

        # separate features and labels
        self.X = data[:, :-1]
        self.y = data[:, -1]

    def plot(self, **kwargs):
        """Plot data."""

        if "color" not in kwargs:
            kwargs["color"] = "b"

        plt.scatter(self.X, self.y, **kwargs)
        plt.xlabel("x", fontsize=16)
        plt.ylabel("y", fontsize=16)
        # plt.show()


# wrapper functions around Data class
def load_data(filename):
    data = Data()
    data.load(filename)
    return data


def plot_data(X, y, **kwargs):
    data = Data(X, y)
    data.plot(**kwargs)


class PolynomialRegression:
    def __init__(self, m=1, reg_param=0):
        """
        Ordinary least squares regression.
        
        Attributes
        --------------------
            coef_   -- numpy array of shape (d,)
                       estimated coefficients for the linear regression problem
            m_      -- integer
                       order for polynomial regression
            lambda_ -- float
                       regularization parameter
        """
        self.coef_ = None
        self.m_ = m
        self.lambda_ = reg_param

    def generate_polynomial_features(self, X):
        """
        Maps X to an mth degree feature vector e.g. [1, X, X^2, ..., X^m].
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,1), features
        
        Returns
        --------------------
            Phi     -- numpy array of shape (n,(m+1)), mapped features
        """

        n, d = X.shape

        ### ========== TODO : START ========== ###
        # part b: modify to create matrix for simple linear model
        # part g: modify to create matrix for polynomial model
        m = self.m_
        if d == m + 1:
            Phi = X
        else:
            new_col = np.ones((n, 1))
            Phi = np.concatenate((new_col, X), axis=1)
            for i in range(2, m + 1):
                new_col = np.power(X, i)
                Phi = np.concatenate((Phi, new_col), axis=1)
        ### ========== TODO : END ========== ###

        return Phi

    def fit_GD(self, X, y, eta=None, eps=0, tmax=10000, verbose=False):
        """
        Finds the coefficients of a {d-1}^th degree polynomial
        that fits the data using least squares batch gradient descent.
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
            eta     -- float, step size
            eps     -- float, convergence criterion
            tmax    -- integer, maximum number of iterations
            verbose -- boolean, for debugging purposes
        
        Returns
        --------------------
            self    -- an instance of self
        """
        if self.lambda_ != 0:
            raise Exception("GD with regularization not implemented")

        if verbose:
            plt.subplot(1, 2, 2)
            plt.xlabel("iteration")
            plt.ylabel(r"$J(\w)$")
            plt.ion()
            plt.show()

        X = self.generate_polynomial_features(X)  # map features
        n, d = X.shape
        eta_input = eta
        self.coef_ = np.zeros(d)  # coefficients
        err_list = np.zeros((tmax, 1))  # errors per iteration

        # GD loop
        start_time = time.time()
        for t in range(tmax):
            ### ========== TODO : START ========== ###
            # part f: update step size
            # change the default eta in the function signature to 'eta=None'
            # and update the line below to your learning rate function
            if eta_input is None:
                eta = 1 / (t + 1)
            else:
                eta = eta_input
            ### ========== TODO : END ========== ###

            ### ========== TODO : START ========== ###
            # part d: update w (self.coef_) using one step of GD
            # hint: you can write simultaneously update all w using vector math
            gradients = np.matmul(X.transpose(), np.matmul(self.coef_, X.transpose()) - y)
            self.coef_ = self.coef_ - 2 * eta * gradients

            # track error
            # hint: you cannot use self.predict(...) to make the predictions
            y_pred = self.predict(X)
            err_list[t] = np.sum(np.power(y - y_pred, 2)) / float(n)
            ### ========== TODO : END ========== ###

            # stop?
            if t > 0 and abs(err_list[t] - err_list[t - 1]) <= eps:
                break

            # debugging
            if verbose:
                x = np.reshape(X[:, 1], (n, 1))
                cost = self.cost(x, y)
                plt.subplot(1, 2, 1)
                plt.cla()
                plot_data(x, y)
                self.plot_regression()
                plt.subplot(1, 2, 2)
                plt.plot([t + 1], [cost], "bo")
                plt.suptitle("iteration: %d, cost: %f" % (t + 1, cost))
                plt.draw()
                plt.pause(0.05)  # pause for 0.05 sec

        end_time = time.time()
        print("Investigating learning rate %f" % eta)
        print("-- number of iterations: %d" % (t + 1))
        print("-- final coefficient: %s " % self.coef_)
        print("-- final cost: %s" % self.cost(X, y))
        print("-- total time takle: %f" % (end_time - start_time))

        return self

    def fit(self, X, y, l2regularize=None):
        """
        Finds the coefficients of a {d-1}^th degree polynomial
        that fits the data using the closed form solution.
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
            l2regularize    -- set to None for no regularization. set to positive double for L2 regularization
                
        Returns
        --------------------        
            self    -- an instance of self
        """

        X = self.generate_polynomial_features(X)  # map features

        ### ========== TODO : START ========== ###
        # part e: implement closed-form solution
        # hint: use np.dot(...) and np.linalg.pinv(...)
        #       be sure to update self.coef_ with your solution
        start_time = time.time()
        w = np.dot(np.dot(np.linalg.pinv(np.dot(X.transpose(), X)), X.transpose()), y)
        self.coef_ = w
        end_time = time.time()
        duration = end_time - start_time

        return self, duration

        ### ========== TODO : END ========== ###

    def predict(self, X):
        """
        Predict output for X.
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
        
        Returns
        --------------------
            y       -- numpy array of shape (n,), predictions
        """
        if self.coef_ is None:
            raise Exception("Model not initialized. Perform a fit first.")

        X = self.generate_polynomial_features(X)  # map features

        ### ========== TODO : START ========== ###
        # part c: predict y
        w = self.coef_
        y = np.matmul(X, w)

        ### ========== TODO : END ========== ###

        return y

    def cost(self, X, y):
        """
        Calculates the objective function.
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
        
        Returns
        --------------------
            cost    -- float, objective J(w)
        """
        ### ========== TODO : START ========== ###
        # part d: compute J(w)
        cost = np.sum(np.square(y - self.predict(X)))
        ### ========== TODO : END ========== ###
        return cost

    def rms_error(self, X, y):
        """
        Calculates the root mean square error.
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
        
        Returns
        --------------------
            error   -- float, RMSE
        """
        ### ========== TODO : START ========== ###
        # part h: compute RMSE
        num_instance, degree = X.shape
        cost = self.cost(X, y)
        error = np.sqrt(cost / num_instance)
        ### ========== TODO : END ========== ###
        return error

    def plot_regression(self, xmin=0, xmax=1, n=50, **kwargs):
        """Plot regression line."""
        if "color" not in kwargs:
            kwargs["color"] = "r"
        if "linestyle" not in kwargs:
            kwargs["linestyle"] = "-"

        X = np.reshape(np.linspace(0, 1, n), (n, 1))
        y = self.predict(X)
        plot_data(X, y, **kwargs)
        plt.show()


######################################################################
# main
######################################################################


def main():
    # load data
    train_data = load_data("regression_train.csv")
    test_data = load_data("regression_test.csv")

    ### ========== TODO : START ========== ###
    # part a: main code for visualizations

    print("Visualizing data...")

    if not os.path.exists("./Graph"):
        os.makedirs("./Graph")

    plt.title("Training dataset")
    plt.grid(True)
    plot_data(train_data.X, train_data.y)
    plt.savefig("./Graph/training_data.png")
    plt.close()
    plt.title("Testing dataset")
    plt.grid(True)
    plot_data(test_data.X, test_data.y, color='red')
    plt.savefig("./Graph/testing_data.png")
    plt.close()

    ### ========== TODO : END ========== ###

    ### ========== TODO : START ========== ###
    # parts b-f: main code for linear regression
    print("Investigating linear regression...")

    # part d
    model = PolynomialRegression()
    model.coef_ = np.zeros(2)
    print("-- model cost with zero weight is: ", model.cost(train_data.X, train_data.y))

    step_size_seq = [1e-4, 1e-3, 1e-2, 0.0407]

    for step_size in step_size_seq:

        model = model.fit_GD(train_data.X, train_data.y, step_size)

    # part e
    model, time = model.fit(train_data.X, train_data.y)
    print("Finding weight with close form solution...")
    print("-- coefficient: %s" % model.coef_)
    print("-- cost: ", model.cost(train_data.X, train_data.y))
    print("-- process finish using %f sec" % time)

    # part f
    print("Investigating proposed learning rate")
    model = model.fit_GD(train_data.X, train_data.y)

    ### ========== TODO : END ========== ###

    ### ========== TODO : START ========== ###
    # parts g-i: main code for polynomial regression
    print("Investigating polynomial regression...")

    # part h
    print("Investigating RMSE with different degree of regression")
    num_instance, _ = train_data.X.shape
    rmse_train_seq = []
    rmse_test_seq =[]
    degree = []

    for m in range(11):

        degree.append(m)
        model = PolynomialRegression(m=m)
        model.coef_ = np.zeros(2)
        model.fit(train_data.X, train_data.y)

        rmse_train = model.rms_error(train_data.X, train_data.y)
        rmse_train_seq.append(rmse_train)
        rmse_test = model.rms_error(test_data.X, test_data.y)
        rmse_test_seq.append(rmse_test)
        print("-- degree: %d, RMSE train: %f, RMSE test: %f" % (m, rmse_train, rmse_test))

    plt.title("Training and testing RMSE with different degree of regression")
    plt.xlabel("Degree of regression")
    plt.ylabel("RMSE")
    plt.grid(True)
    plt.plot(degree, rmse_train_seq, marker='x', label='Training RMSE')
    plt.plot(degree, rmse_test_seq, marker='o', label='Testing RMSE')
    plt.legend(loc="upper left")
    plt.savefig("./Graph/RMSE.png")


    ### ========== TODO : END ========== ###

    print("Done!")


if __name__ == "__main__":
    main()
