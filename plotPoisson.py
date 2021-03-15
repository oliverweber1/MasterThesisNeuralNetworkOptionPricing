# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 18:12:35 2020

@author: Oliver Weber
"""


import matplotlib.pyplot as plt
from OneDimProcesses import PoissonProcess, CompensatedPoissonProcess

# use LaTeX fonts in the plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

lam = 5
T = 1
n = 3

poi = PoissonProcess(lam, T)

cpoi = CompensatedPoissonProcess(lam, T)

fig, ax = plt.subplots(1, 2, figsize=(6,2.5))
for _ in range(n):
    ax[0].plot(poi.time, poi.generatePath().T, linewidth='0.2', linestyle='dashed', marker='o', markersize='0.2')
    ax[1].plot(poi.time, cpoi.generatePath().T, linewidth='0.2', linestyle='dashed', marker='o', markersize='0.2')
ax[0].set_title(r'Poisson Process', fontsize=10)
ax[1].set_title('Compensated Poisson Process', fontsize=10)
plt.savefig('C:\\Users\\oli-w\\OneDrive\\Uni\\Master Thesis\\LaTeX Template\\poissonPath.png', dpi=300)
# best result: name.pdf but made file too slow in this case

