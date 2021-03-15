# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 16:45:13 2021

@author: Oliver Weber
"""

from OptionValuation import EuropeanCall
from OneDimProcesses import JumpDiffusionProcess
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sp

def getCallPricesStrike(strikes, nPaths, StockModel):
    prices = []
    for k in strikes:
        call = EuropeanCall(k, StockModel=StockModel)
        prices.append(call.MonteCarloPrice(N=nPaths))
    return prices

r = 0.05
sig = 0.25
mu = -0.5 * sig ** 2

# Black Scholes Model
BS = JumpDiffusionProcess(lam = 0, mu=mu, r=r, sigma=sig)

# Black Scholes + Jumps
lam = 1
jumps = sp.norm(scale=0.3)
JD = JumpDiffusionProcess(lam = lam, mu=mu, r=r, sigma=sig, jumpSizeRV=jumps)

# Black Scholes + Large Jumps
jumps2 = sp.norm(scale=0.7)
JDLarge = JumpDiffusionProcess(lam = lam, mu=mu, r=r, sigma=sig, jumpSizeRV=jumps2)

# Black Scholes + Frequent Jumps
lam2 = 3
JDFreq = JumpDiffusionProcess(lam = lam2, mu=mu, r=r, sigma=sig, jumpSizeRV=jumps)

# Black Scholes + Large Frequent Jumps
JDLargeFreq = JumpDiffusionProcess(lam = lam2, mu=mu, r=r, sigma=sig, jumpSizeRV=jumps2)

minStrike = 0
maxStrike = 2.5
stepSize = 0.1
strikes = np.arange(minStrike, maxStrike + stepSize, stepSize)

N = 100000

BSprices = getCallPricesStrike(strikes, N, BS)
print('BS prices done')
JDprices = getCallPricesStrike(strikes, N, JD)
print('JD prices done')
JDLprices = getCallPricesStrike(strikes, N, JDLarge)
print('JDL prices done')
JDFprices = getCallPricesStrike(strikes, N, JDFreq)
print('JDF prices done')
JDLFprices = getCallPricesStrike(strikes, N, JDLargeFreq)
print('JDLF prices done')

plt.figure(figsize=(10,5))
plt.plot(strikes, BSprices, label='Black-Scholes')
plt.plot(strikes, JDprices, label='Jump Diffusion')
plt.plot(strikes, JDLprices, label='Jump Diff (Large Jumps)')
plt.plot(strikes, JDFprices, label='Jump Diff (More Jumps)')
plt.plot(strikes, JDLFprices, label='Jump Diff (Large + More Jumps)')
plt.xlabel('Strike')
plt.ylabel('Call Option Price')
plt.legend()


