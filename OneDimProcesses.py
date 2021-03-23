"""
Created on Wed Dec 30 17:36:36 2020

@author: Oliver Weber
"""

import numpy as np
import scipy.stats as sp
from MultiDimProcesses import MultiDimProcess, CorrelatedBrownianMotion

class PoissonProcess(MultiDimProcess):
    
    def __init__(self, lam, T=1, N=10000, x0=0):
        super().__init__(T, N, 1, x0)
        self.lam = lam
        self.jumpNumberRV = sp.poisson(mu = lam * T)
        
    def generatePath(self):
        numberOfJumps = self.jumpNumberRV.rvs()
        jumpTimes = np.append(sorted(sp.uniform.rvs(scale = self.T, size = numberOfJumps)), self.T+1)
        jumpValues = np.arange(self.x0, self.x0 + numberOfJumps + 1)
        path = []
        jumpIndex = 0
        for i in range(len(self.time)):
            if self.time[i] < jumpTimes[jumpIndex]:
                path.append(jumpValues[jumpIndex])
            else:
                jumpIndex += 1
                path.append(jumpValues[jumpIndex])
        return np.array(path)
    
    def getValue(self, t=None):
        if t is None:
            t = self.T
        return self.x0 + sp.poisson(mu=self.lam*t)
    
    def plotPath(self, size=(10,5)):
        super().plotPath('Poisson Process', size)
        
class CompoundPoissonProcess(MultiDimProcess):
    
    def __init__(self, lam, T=1, N=10000, x0=0, jumpSizeRV=sp.norm):
        super().__init__(T, N, 1, x0)
        self.lam = lam
        self.jumpNumberRV = sp.poisson(mu = lam * T)
        self.jumpSizeRV = jumpSizeRV
        
    def generatePath(self):
        numberOfJumps= self.jumpNumberRV.rvs()
        jumpTimes = np.append(sorted(sp.uniform.rvs(scale = self.T, size = numberOfJumps)), self.T+1)
        jumpValues = np.cumsum(np.append(self.x0, self.jumpSizeRV.rvs(size = numberOfJumps)))
        path = []
        jumpIndex = 0
        for i in range(len(self.time)):
            if self.time[i] < jumpTimes[jumpIndex]:
                path.append(jumpValues[jumpIndex])
            else:
                jumpIndex += 1
                path.append(jumpValues[jumpIndex])
        return np.array(path)
    
    def getValue(self, t=None):
        if t is None:
            t = self.T
        return self.x0 + sum(self.jumpSizeRV.rvs(size=sp.poisson(mu=self.lam*t).rvs()))
    
    def plotPath(self, size=(10,5)):
        super().plotPath('Poisson Process', size)
        
        
class CompensatedCompoundPoissonProcess(CompoundPoissonProcess):
    
    def __init__(self, lam, T=1, N=10000, x0=0, jumpSizeRV=sp.norm):
        super().__init__(lam, T, N, x0, jumpSizeRV)
        self.lam = lam
        self.meanJumpSize = jumpSizeRV.mean()
        
    def generatePath(self):
        return super().generatePath() - self.lam * self.time * self.meanJumpSize
    
    def plotPath(self, size=(10,5)):
        MultiDimProcess.plotPath(self, 'Compensated Poisson Process', size)
        
class CompensatedPoissonProcess(PoissonProcess):
    
    def __init__(self, lam, T=1, N=10000, x0=0):
        super().__init__(lam, T, N, x0)
        self.lam = lam
        
    def generatePath(self):
        return super().generatePath() - self.lam * self.time
    
    def plotPath(self, size=(10,5)):
        MultiDimProcess.plotPath(self, 'Compensated Poisson Process', size)
        
        
class JumpDiffusionProcess(MultiDimProcess):
    
    def __init__(self, lam, T=1, N=10000, x0=1, jumpSizeRV=sp.norm, mu=0, r=0, sigma=1):
        super().__init__(T, N, 1, x0)
        self.r = r
        self.BrownianPart = CorrelatedBrownianMotion(T, N, 1, mu=mu+r, sigma=sigma)
        self.PoissonPart = CompoundPoissonProcess(lam, T, N, jumpSizeRV=jumpSizeRV)
        
    def generatePath(self):
        return self.x0 * np.exp(self.BrownianPart.generatePath() + self.PoissonPart.generatePath())
    
    def getValue(self, t=None):
        if t is None:
            t = self.T
        return self.x0 * np.exp(self.BrownianPart.getValue(t) + self.PoissonPart.getValue(t))
    
    def plotPath(self, size=(10,5)):
        MultiDimProcess.plotPath(self, 'Jump Diffusion Process', size)
        
        
        