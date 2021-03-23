# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 12:30:46 2021

@author: Oliver Weber
"""

import scipy.stats as sp
import numpy as np

def blackScholesCallPrice(x0, r, sigma, T, K):
    
    # returns analytical value of call option price in 1-dim Black Scholes model with
    # strike K, maturity T, interest rate r and volatility sigma of the underlying
    
    d1 = (np.log(x0 / K) + (r + sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    return (x0 * sp.norm.cdf(d1, 0.0, 1.0) - \
            K * np.exp(-r * T) * sp.norm.cdf(d2, 0.0, 1.0))
    
