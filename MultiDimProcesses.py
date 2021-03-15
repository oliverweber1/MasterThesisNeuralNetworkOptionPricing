
"""
Created on Mon Nov  2 18:31:54 2020

@author: Oliver Weber
"""

import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt

class MultiDimProcess:
    
    # Abstract base class for multi-dimensional stochastic processes
    # 
    # Attributes:
    # T: Time horizon - Process lives on [0,T] (Double > 0)
    # N: Steps for time discretization of paths (Integer > 0)
    # time: Time discretization of N equally spaced time points between 0 and T
    # d: Dimension (Integer > 0)
    # x0: Starting value (dx1 numpy array)
    #
    # Constructor Input:
    # T (Double > 0)
    # N (Integer > 0)
    # d (Integer > 0)
    # x0: (Double) will be broadcasted to dx1 numpy array, or (dx1 numpy array)
    
    def __init__(self, T, N, d, x0):
        self.T = T
        self.N = N
        self.time = np.linspace(0, T, N)
        self.d = d
        self.x0 = np.ones((d,1)) * x0 if np.isscalar(x0) else x0
        
    def generatePath(self):
        raise NotImplementedError('Must override generatePath')
        
    def getValue(self):
        raise NotImplementedError('Must override getValue')
        
    def plotPath(self, processName, size):
        # plots one sample path for each component of the process
        plt.figure(figsize=size)
        plt.plot(self.time, self.generatePath().T)
        plt.xlabel('$t$')
        plt.ylabel('$S_t$')
        plt.title('Sample path(s) of a {}-dimensional '.format(self.d) + processName)
        
        
class CorrelatedBrownianMotion(MultiDimProcess):
    
    # Correlated Brownian Motion with drift and scaled diffusion:
    #
    #   X_t = mu*t + sigma*F*W_t
    #
    # with drift vector mu, diffusion vector sigma, factor matrix F obtained from Cholesky decomposition
    # of a correlation matrix and d-dimensional standard Brownian motion W
    #
    # Attributes:
    # F: Factor matrix that induces correlations between the Brownian components (dxd numpy array), Default: Identity
    # mu: Drift vector (dx1 numpy array), Default: vector of zeros
    # sigma: Diffusion vector (dx1 numpy array), Default: vector of ones
    #
    # Constructor Input:
    # Input for MultiDimProcess
    # Corr: Correlation matrix (dxd numpy array) or correlation coefficient (Double in ]-1,1[) 
    # mu: Drift vector (dx1 array)
    # sigma: Diffusion vector (dx1 array)
    
    def __init__(self, T=1, N=10000, d=1, x0=0, Corr=None, mu=None, sigma=None):
        super().__init__(T, N, d, x0)
        if Corr is None:
            self.Factors = np.eye(d)
        else:
            self.correlate(Corr)
        self.mu = np.zeros((d,1)) if mu is None else np.reshape(mu, (d,1))
        self.sigma = np.ones((d,1)) if sigma is None else np.reshape(sigma, (d,1))
        
        
    def correlate(self, Corr):
        # induces correlation to the BM
        # if correlation coefficient is provided, create correlation matrix first
        if np.isscalar(Corr):
            Corr = np.ones((self.d,self.d)) * Corr + np.diag([1-Corr] * self.d)
        self.Factors = np.linalg.cholesky(Corr)
        
    def generatePath(self):
        # generate a sample path (one for each dimension)
        # 1. Samples N-1 random variables ~N(0,dt) where dt is the step size and sums them (standard BM)
        # 2. add start value and drift, multiply with correlation factors, scale by sigma
        dt = self.T / (self.N - 1)
        increments = np.append(np.zeros((self.d,1)), sp.norm.rvs(scale=np.sqrt(dt), size=(self.d,self.N-1)), axis=1)
        return self.x0 + self.mu * self.time + self.sigma * (self.Factors @ np.cumsum(increments,axis=1))
    
    def getValue(self, t=None):
        # returns value for a given timepoint t (i.e. a random variable ~ N_d(x0+t*mu,t*sigma^2) with Corr)
        # Warning: simulating many timepoints doesn't give a BM! (no corr between timepoints)
        # only useful if only one timepoint is needed, i.e. for European option valuation
        if t is None:
            t = self.T
        return  self.x0 + self.mu * t + self.sigma * self.Factors @ sp.norm.rvs(scale=np.sqrt(t), size=(self.d,1))
    
    def plotPath(self, size=(10,5)):
        super().plotPath('Brownian Motion', size)
        

class GeometricBrownianMotion(MultiDimProcess):
    
    # Generates a geometric Brownian Motion under the risk neutral measure, i.e.
    # a process that solves
    #
    # dS_t = r*S_t dt + sigma*S_t dW_t
    #
    # The solution is
    #
    # S_t = S_0 * exp(r - 1/2*sigma^2) * t + sigma*W_t)
    #
    # for starting value S_0, interest rate r, and correlated Brownian Motion W.
    
    def __init__(self, T=1, N=10000, d=1, r=0, sigma=None, x0=1, Corr=None):
        super().__init__(T, N, d, x0)
        self.r = r
        self.sigma = np.ones((self.d,1)) if sigma is None else np.reshape(sigma, (self.d,1))
        self.BrownianDriver = CorrelatedBrownianMotion(T, N, d, 0, Corr)
        
    def generatePath(self):
        return self.x0 * np.exp((self.r - 0.5 * self.sigma ** 2) * self.time + self.sigma * self.BrownianDriver.generatePath())
    
    def getValue(self, t=None):
        if t is None:
            t = self.T
        return self.x0 * np.exp((self.r - 0.5 * self.sigma ** 2) * t + self.sigma * self.BrownianDriver.getValue(t))
    
    def plotPath(self, size=(10,5)):
        super().plotPath('Geometric Brownian Motion', size)

    
        

        


