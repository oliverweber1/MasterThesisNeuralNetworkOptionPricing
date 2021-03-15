# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 12:16:46 2020

@author: Oli
"""
from MultiDimProcesses import CorrelatedBrownianMotion

import matplotlib.pyplot as plt
# use LaTeX fonts in the plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

bm = CorrelatedBrownianMotion(d=5)
plt.plot(bm.time, bm.generatePath().T)

# set labels (LaTeX can be used)
plt.title(r'\textbf{Example Title}', fontsize=11)
plt.xlabel(r'$t$', fontsize=11)
plt.savefig('my_eps_plot.pdf')