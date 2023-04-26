import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class quasarView(object):
    """Class built to apply regression to the collection of Quasar data with the following properties:

    Attributes:
        lamb = Wavelengths
        flux = Flux Measurements
        theta = parameters
        h = hypothesis values
        x = input features
        y = target features
        J = cost
        w = weight
        W = diagnol matrix of w to use for cost function
    """

    def __init__(self, data, n):
        """initializes attributes
        """
        self.df = data
        self.theta = []
        self.h = []
        self.x = np.array(data.columns.values.astype(float))
        self.X = np.vstack((np.ones(self.x.shape), self.x)).T
        self.y = data.head(n).values.T
        self.Y = data.values.T
        self.J = []
        self.w = []
        self.weights = []

    # def localWeight(self, x_i, x, tau = 5):
    #     """generate locally fitted weights
    #     """
    #     self.w = []
    #     self.weights = []
        
    #     for i in range(len(x)):
    #         self.w.append(np.exp((-(x_i - x[i])[:,1]**2)/(2*tau**2)))
    #     return np.diag(self.w)
    
    def localWeight(self, x, x_i, tau=5):
        return np.diag(np.exp(-((x_i-x)[:,1]**2)/(2*tau**2)))
        
    # def normalEq(self, weight = False):
    #     """"uses normal equation to implement an unweighted linear fit onto provided training example
    #     """
    #     if weight is False:
    #         self.theta = np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T.dot(self.y))
    #         self.h = self.X.dot(self.theta)
    #         return self.h
    #     else:
    #         for x_i in self.X:
    #             w = self.localWeight(x_i[1], self.X[:,1], tau = 5)
    #             self.theta = np.linalg.inv(self.X.T.dot(w).dot(self.X)).dot(self.X.T.dot(w)).dot(self.y)
    #             # self.h.append(float(x_i.dot(self.theta)))

    #         return self.theta
        
    def normalEQ(self, x, y, weight=None):
        """"uses normal equation to implement an unweighted linear fit onto provided training example 
        """
        if weight is None:
            return np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
        else:
            return np.linalg.inv(x.T.dot(weight).dot(x)).dot(x.T).dot(weight).dot(y)
        
    def smoother(self, tau = 5):
        """applies smoothing to input data based on techniques used above
        """
        x = self.X
        y_in = self.Y

        self.h = np.zeros(y_in.shape)
        for i in range(y_in.shape[1]):
            y = y_in[:,i]
            for j, x_j in enumerate(x):
                w = self.localWeight(x, x_j,tau)
                theta = self.normalEQ(x, y, w)
                self.h[j][i] =  theta.T.dot(x_j[:,np.newaxis]).ravel()[0]
        return self.h
                
    def sqDist(self, f_1, f_2):
        """computes squared distance between new & old data points 
        takes 2 spectrums (f_1 & f_2) as inputs and outputs a scalar
        """
        return np.sum((f_1-f_2[:,np.newaxis])**2,0)
    
    def kernel(self, t):
        return np.maximum(1-t,0)
    
    def nearestNeighbors(self, deltas, k):
        """returns k indices of the training set that are closest to f_r
        """
        idx = np.argsort(deltas)[:k+1]
        return idx[:k]

    def funcRegression(self, y_train, y_test, lyman_alpha):
        """Constructs the cuntional regression estimate for each spectrum
        """

        # Slice data according to our Lyman-alpha
        x = self.X[:,1]
        self.y_train_left = y_train[x < lyman_alpha,:]
        self.y_train_right = y_train[x >= lyman_alpha+100,:]
        self.y_test_left = y_test[x < lyman_alpha,:]
        self.y_test_right = y_test[x >= lyman_alpha+100,:]

        # Format our estimate matrix to store results
        self.f_hat_left = np.zeros(self.y_test_left.shape)

        # Compute estimate function
        for i in range(self.y_test_right.shape[1]):
            deltas = self.sqDist(self.y_train_right, self.y_test_right[:,i])
            idx = self.nearestNeighbors(deltas,3)

            h = np.max(deltas) 
            weights = self.kernel(deltas/h)[idx]
            
            f_hat_num = np.sum(self.y_train_left[:,idx]*weights,1)
            f_hat_den = np.sum(weights)
            self.f_hat = f_hat_num/f_hat_den
            self.f_hat_left[:,i] = self.f_hat
        return self.f_hat_left

    def plot_unweightedFit(self, theta):
        """plot generated normal fit
        """
        ax1 = sns.regplot(x=self.X[:,1], y=self.y)
        ax1.set(xlabel="Wavelength", ylabel='Flux')
        plt.plot(self.X[:,1], self.h)
        plt.show()

    def plot_weightedFit(self, h):
        """plot generated normal fit
        """
        ax1 = sns.regplot(x=self.X[:,1], y=self.Y)
        ax1.set(xlabel="Wavelength", ylabel='Flux')
        plt.plot(self.X[:,1], h)
        plt.show()

    def plot_functionalRegression(self, y_train, y_test, h):
        y_train_left = y_train[self.X[:,1] < 1200,:]
        y_test_left = y_test[self.X[:,1] < 1200,:]

        plt.plot(self.X[self.X[:,1] < 1200,1],h)
        
        error = np.mean(np.sum((h - y_train_left)**2,0))
        plt.show()

        print(error)

   





def main():
    df_train = pd.read_csv(r'C:\Users\Richard\Downloads\quasar_train.csv')
    df_test = pd.read_csv(r'C:\Users\Richard\Downloads\quasar_test.csv')

    train_fit = quasarView(df_train, 1)
    test_fit = quasarView(df_test, 1)

    y_train = train_fit.smoother()
    y_test = test_fit.smoother
    # y_test = test_fit.normalEq(weight = True)

    func_regression = train_fit.funcRegression(y_train, y_test, 1200)
    
    train_fit.plot_functionalRegression(y_train, y_test, func_regression)

    # 

    return 0


if __name__ == "__main__":
    main()