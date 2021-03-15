# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 15:24:22 2020

@author: Oli
"""


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from MultiDimProcesses import CorrelatedBrownianMotion

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

bm = CorrelatedBrownianMotion(d=5)
bm.plotPath(size=(6,3))
plt.savefig('example.pgf')

