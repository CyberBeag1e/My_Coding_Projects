import math
import numpy as np
import pandas as pd
from matplotlib import rcParams
import matplotlib.pyplot as plt
import scipy.optimize as sco


class Portfolio:
    def __init__(self):
        """
        portfolio: numpy.ndarray, array of daily return of selected stocks
        num: int, mumber of stocks involved
        rf: float, risk-free interest rate
        rtn: numpy.ndarray, array of average return of stocks involved
        cov: numpy.ndarray, covariance matrix
        corr: numpy.ndarray, correlation matrix
        """
        self._portfolio = None
        self._num = None
        self._rf = None
        self._rtn = None
        self._cov = None

    def addStocks(self, data, type="return"):
        """
        data: prefer pandas.dataframe, the shape should be (m,n),
              where m denotes the number of periods, and n denotes the number of stocks

        type: 'return' -> stock returns data
              'price' -> stock prices data, needs to convert to return type data
        """
        stocks = np.array(data)
        if type == "price":
            stocks = self.rtnCal(stocks)

        self._portfolio = stocks
        self._num = self._portfolio.shape[1]

    ## Specify risk-free interest rate
    def setInterest(self, interest_rate):
        self._rf = float(interest_rate)

    ## Convert stock prices to stock returns
    ## If the shape of original data is (m,n), the shape of output will be (m-1,n)
    def rtnCal(self, data):
        array1 = np.array(data)[1:, ...]
        array2 = np.roll(np.array(data), [1 for i in range(data.shape[1])])[1:, ...]
        rtn = (array1 - array2) / array2

        return rtn

    ## Calculate average return
    ## If the shape of original data is (m,n), the shape of output will be (n,)
    def avgRtn(self):
        self._rtn = np.mean(self._portfolio, axis=0)
        return self._rtn

    ## Calculate covariance matrix
    ## If the shape of original data is (m,n), the shape of output will be (n,n)
    def covCal(self):
        self._cov = np.cov(self._portfolio.T)
        return self._cov

    def getCov(self):
        return self._cov

    def getPortfolio(self):
        return self._portfolio

    ## Calculate the return and volatility of the portfolio, given the weights of investment
    def portfolioCal(self, weights):
        self.avgRtn()
        self.covCal()
        weights = np.array(weights)
        portfolio_rtn = np.dot(weights.T, self._rtn)
        portfolio_std = math.sqrt(np.dot(weights.T, np.dot(self._cov, weights)))

        return {
            "return": portfolio_rtn,
            "deviation": portfolio_std
        }

    ## Calculate the Sharpe Ratio of the portfolio, given the weights of investment
    def sharpeRatio(self, weights):
        if self._rf is not None:
            rtn = self.portfolioCal(weights)['return']
            std = self.portfolioCal(weights)['deviation']
            return (rtn - self._rf) / std
        else:
            ## you must set interest rate first
            raise ValueError("Interest rate not defined!")

    ## Calculate the minimum possible standard deviation
    def minStd(self, tgt=None):
        """
        tgt: target return of the portfolio. If not specified, the method will find
             the minimum standard deviation of all possible portfolio
        """
        if tgt is None:
            """
            Set constraints:
            1. Sum of weights equals 1
            2. Maximum weight not greater than 1
            3. All weights greater than 0
            """
            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                    {'type': 'ineq', 'fun': lambda x: 1 - np.max(x)},
                    {'type': 'ineq', 'fun': lambda x: np.min(x)})
        else:
            """
            Set constraints:
            1. Target return satisfied
            """
            cons = ({'type': 'eq', 'fun': lambda x: self.portfolioCal(x)['return'] - tgt},
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                    {'type': 'ineq', 'fun': lambda x: 1 - np.max(x)},
                    {'type': 'ineq', 'fun': lambda x: np.min(x)})

        bnds = tuple((0, 1) for x in range(self._num))
        ## Optimization process, SLSQP represents Sequential Least Squares Programming
        ## Returns optimal weights array
        optv = sco.minimize(lambda x: self.portfolioCal(x)['deviation'],
                            self._num * [1. / self._num, ],
                            method='SLSQP', bounds=bnds, constraints=(cons))

        rtnv = self.portfolioCal(optv.x)['return']
        stdv = self.portfolioCal(optv.x)['deviation']
        wtv = optv.x

        return {
            'return': rtnv,
            'deviation': stdv,
            'weights': wtv
        }

    def optimize(self):  ## in terms of minimum Sharpe ratio
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'ineq', 'fun': lambda x: 1 - np.max(x)},
                {'type': 'ineq', 'fun': lambda x: np.min(x)})

        bnds = tuple((0, 1) for x in range(self._num))
        opts = sco.minimize(lambda x: -self.sharpeRatio(x),
                            self._num * [1. / self._num, ],
                            method='SLSQP', bounds=bnds, constraints=(cons))

        rtns = self.portfolioCal(opts.x)['return']
        stds = self.portfolioCal(opts.x)['deviation']
        wts = opts.x

        ## If the overall performance of the portfolio is worse than T-bills,
        ## the optimal investment desicion should be buying T-bills will all funds
        if rtns >= self._rf:
            return {
                'return': rtns,
                'deviation': stds,
                'weights': wts
            }
        else:
            return {
                'return': self._rf,
                'deviation': 0,
                'weights': None
            }

    ## Get the efficient frontier of the portfolio
    def generateFrontier(self):
        target_rtn = []
        target_std = []
        target_wt = []

        rtn_min = self.minStd()['return']

        ## Set a sequence of target returns by interpolation
        for tgt in np.linspace(rtn_min - self.scaler(rtn_min),
                               rtn_min + self.scaler(rtn_min), 200):
            opt = self.minStd(tgt=tgt)
            target_rtn.append(tgt)
            target_std.append(opt['deviation'])
            target_wt.append(opt['weights'])

        return {
            'returns': np.array(target_rtn),
            'deviations': np.array(target_std),
            'weights': np.array(target_wt)
        }

    ## Simulation method to generate the efficient frontier of the portfolio
    def simulation(self, n):
        num = self._num
        returns = []
        deviations = []

        max_sharpe = 0
        rtn_max_sharpe = self._rf
        std_max_sharpe = 0
        wt_max_sharpe = None

        for i in range(n):
            ## Generate random weights that sum up to 1
            randNums = np.random.rand(num)
            randWeights = randNums / np.sum(randNums)

            rtn_temp = self.portfolioCal(randWeights)['return']
            std_temp = self.portfolioCal(randWeights)['deviation']
            returns.append(rtn_temp)
            deviations.append(std_temp)

            sharpe = (rtn_temp - self._rf) / std_temp
            if sharpe > max_sharpe:
                max_sharpe = sharpe
                rtn_max_sharpe = rtn_temp
                std_max_sharpe = std_temp
                wt_max_sharpe = randWeights

        return {
            'returns': np.array(returns),
            'deviations': np.array(deviations),
            'max_sharpe': max_sharpe,
            'rtn_max_sharpe': rtn_max_sharpe,
            'std_max_sharpe': std_max_sharpe,
            'wt_max_sharpe': wt_max_sharpe
        }

    ## Determine the scale of a number
    def scaler(self, number):
        scale = int(f"{number:.2e}"[-1])
        return 1. / pow(10, scale)

    ## Plot the simulation results and efficient frontier
    def plot(self, n=10000):
        sim = self.simulation(n)
        frontier = self.generateFrontier()
        optimal = self.optimize()

        rcParams['font.family'] = 'Times New Roman'
        fig, axes = plt.subplots(1, 2)
        fig.set_figheight(10)
        fig.set_figwidth(20)

        for ax in axes:
            ax.set_xlim(0, math.sqrt(np.amax(self._cov)))
            ax.set_xlabel('Standard Deviation', fontsize=12)
            ax.set_ylabel('Expected Return', fontsize=12)

        axes[0].set_title(f"Simulation", fontsize=18)
        axes[1].set_title('Efficient Frontier', fontsize=18)

        axes[0].scatter(sim['deviations'], sim['returns'], s=1)
        axes[0].plot(sim['std_max_sharpe'], sim['rtn_max_sharpe'],
                     '*', markersize=5, c="red",
                     label='Tangency Portfolio')

        axes[0].plot([0, sim['std_max_sharpe']], [self._rf, sim['rtn_max_sharpe']],
                     '-', linewidth=1,
                     label='Efficient Portfolio of T-Bills and Tangency Portfolio')

        axes[1].plot(frontier['deviations'], frontier['returns'],
                     '-', linewidth=1,
                     label="Efficient Frontier")

        axes[1].plot(optimal['deviation'], optimal['return'],
                     '*', markersize=5, c="red",
                     label="Portfolio with minimum Sharpe ratio")

        axes[1].plot([0, optimal['deviation']], [self._rf, optimal['return']],
                     '-', linewidth=1,
                     label='Efficient Portfolio of T-Bills and Tangency Portfolio')

        axes[0].legend(loc="upper right", prop={'family': "Times New Roman"}, fontsize=9)
        axes[1].legend(loc="upper right", prop={'family': "Times New Roman"}, fontsize=9)

        plt.show()


def main(filepath):
    ## Read File
    df_stock = pd.read_csv(filepath)
    print(f"File read in {filepath}")
    # or df_return=pd.read_csv(filepath)

    portfolio = Portfolio()
    portfolio.addStocks(df_stock, type='price')
    print("Data added")
    # or: portfolio.addStocks(df_return)

    portfolio.setInterest(0.01)
    portfolio.plot()


filepath = "Files/StockData.csv"
main(filepath)