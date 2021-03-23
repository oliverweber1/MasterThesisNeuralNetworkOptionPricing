# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 14:38:18 2021

@author: Oliver Weber
"""

import numpy as np
import matplotlib.pyplot as plt

# use LaTeX fonts in the plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif')



x = np.linspace(0,5,200)
y_pred = lambda x : 0.2 * np.sin(4 * x) + 1 - np.exp(-x)
y_pred2 = lambda x : 0.2 + 0.2 * x
y_true = lambda x : 1 - np.exp(-x)

x_train = np.array([0.3, 0.6, 1.2, 1.8, 2.0, 2.8, 3.2, 3.6, 4.2, 4.5])
x_test = np.array([0.1, 0.4, 1, 1.8, 2.2, 2.8, 3.4, 3.7, 4.4, 4.7])
y_train = y_pred(x_train)
y_test = y_true(x_test) + np.random.normal(0, 0.1, x_test.shape)


fig, ax = plt.subplots(1, 3, figsize=(6,2))
ax[0].plot(x, y_pred(x), color='black', linewidth=0.5, label='Hypothesis')
ax[1].plot(x, y_true(x), color='black', linewidth=0.5, label='Hypothesis')
ax[2].plot(x, y_pred2(x), color='black', linewidth=0.5, label='Hypothesis')
for a in ax:
    a.plot(x_train, y_train, 'o', color='red', markersize=2, label='Training Data')
    a.plot(x_test, y_test, 'o', color='blue', markersize=2, label='Test Data')
    a.set_xlabel('$x$', fontsize=8)
    a.tick_params(axis='both', which='major', labelsize=6)
handles, labels = ax[2].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.1), fontsize=10)
ax[0].set_title('Overfit', fontsize=10)
ax[1].set_title('Good Fit', fontsize=10)
ax[2].set_title('Underfit', fontsize=10)
ax[0].set_ylabel('$y$', fontsize=8)
plt.tight_layout()


plt.savefig('C:\\Users\\oli-w\\OneDrive\\Uni\\Master Thesis\\LaTeX Template\\overunderfit.png', dpi=300, bbox_inches='tight')
