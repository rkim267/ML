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

    def __init__(self, data, m, n):
        """initializes attributes
        """
        self.df = data
        self.theta = []
        self.h = []
        self.x = np.array(data.columns.values.astype(float))
        self.X = np.vstack((np.ones(self.x.shape), self.x)).T
        self.y = data.head(n).values.T
        self.J = []
        self.w = []
        self.weights = []

    def localWeight(self, x_i, x, tau = 5.0):
        """generate locally fitted weights
        """
        self.w = []
        self.weights = []
        
        for i in range(len(x)):
            self.w.append(np.exp((-(x_i - x[i])**2)/(2*tau**2)))
            #self.weights.append(self.w)       

        return np.diag(self.w)
        
    def normalEq(self, weight = False):
        """"uses normal equation to implement an unweighted linear fit onto provided training example
        """
        if weight is False:
            self.theta = np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T.dot(self.y))
            self.h = self.X.dot(self.theta)
            return self.h
        else:
            for x_i in self.X:
                w = self.localWeight(x_i[1], self.X[:,1], tau = 5)
                self.theta = np.linalg.inv(self.X.T.dot(w).dot(self.X)).dot(self.X.T.dot(w)).dot(self.y)
                self.h.append(float(x_i.dot(self.theta)))

            return self.h
                                    
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
        ax1 = sns.regplot(x=self.X[:,1], y=self.y)
        ax1.set(xlabel="Wavelength", ylabel='Flux')
        plt.plot(self.X[:,1], h)
        plt.show()
   





def main():
    df_train = pd.read_csv(r'C:\Users\Richard\Downloads\quasar_train.csv')

    train_fit = quasarView(df_train, 1, 1)

    localFit = train_fit.normalEq(weight = True)

    train_fit.plot_weightedFit(localFit)

    # w = train_fit.localWeight()

    # theta = train_fit.normalEq()

    # train_fit.plot_Fit(theta)

    return 0


if __name__ == "__main__":
    main()