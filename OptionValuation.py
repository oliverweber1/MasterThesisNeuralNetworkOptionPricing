"""
Created on Mon Nov  2 18:51:04 2020

@author: Oliver Weber
"""

from MultiDimProcesses import GeometricBrownianMotion
import numpy as np

class EuropeanOption:
    
    def __init__(self, StockModel):
        self.StockModel = StockModel
        
    
    def payoffFunction(self):
        raise NotImplementedError('Must override payoffFunction')
        

class EuropeanCall(EuropeanOption):
    
    def __init__(self, K, d=0, StockModel=GeometricBrownianMotion()):
        super().__init__(StockModel)
        self.d = d
        self.K = K
        
    def payoffFunction(self, x):
        return np.maximum(x - self.K, 0)
    
    def MonteCarloPrice(self, N=10000, t=None):
        # t : time to Maturity
        if t is None:
            t = self.StockModel.T
        x = np.array([self.StockModel.getValue(t)[self.d] for _ in range(N)])
        return np.exp(-self.StockModel.r * t) * (self.payoffFunction(x).mean())
    
class EuropeanPut(EuropeanOption):
    
    def __init__(self, K=None, d=0, StockModel=GeometricBrownianMotion()):
        super().__init__(StockModel)
        self.d = d
        self.K = K
        
    def payoffFunction(self, x, K=None):
        if K is None:
            K = self.K
        return np.maximum(self.K - x, 0)
    
    def MonteCarloPrice(self, N=10000, t=None, K=None):
        # t : time to Maturity
        if t is None:
            t = self.StockModel.T
        x = np.array([self.StockModel.getValue(t)[self.d] for _ in range(N)])
        return np.exp(-self.StockModel.r * t) * (self.payoffFunction(x, K).mean())
    
        

