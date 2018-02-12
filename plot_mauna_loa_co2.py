#!/usr/bin/env python

"""
Get Mauna Loa CO2 data and make a plot using gaussian processes machine learning

The kernel is composed of several terms:
- a long term, smooth rising trend, explained by an RBF kernel (smooth)
- a seasonal component, explained by the periodic ExpSineSquared kernel with a
  fixed periodicity of 1 year.
- smaller, medium term irregularities, explained by a RationalQuadratic kernel.
  component.
- a noise term, consisting of an RBF kernel contribution, and a WhiteKernel
  contribution for the white noise.

from http://scikit-learn.org
"""

__author__ = "Martin De Kauwe"
__version__ = "1.0 (12.02.2018)"
__email__ = "mdekauwe@gmail.com"

from urllib.request import urlretrieve
import matplotlib.pyplot as plt
from pathlib import Path
import os
import pandas as pd
import numpy as np
import sys
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ExpSineSquared
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import WhiteKernel

address = 'ftp://aftp.cmdl.noaa.gov/products/trends/co2/'
fname = "co2_mm_mlo.txt"
if not Path(fname).exists():
    urlretrieve(os.path.join(address, fname), fname)

hdr = ["year", "month", "decimal_date", "average", \
       "interpolated", "trend", "days"]
df = pd.read_csv(fname, comment='#', delim_whitespace=True,
                 names=hdr, na_values=[-99.99, -1])
df = df[df['year'].notnull()] # drop NaNs

date = np.array(df.decimal_date)

date = date[df.interpolated > 0]
co2 = df.interpolated[df.interpolated > 0].values

X = np.array(date, ndmin=2).T
y = co2

# Kernel with parameters given in GPML book

# long term smooth rising trend
k1 = 66.0**2 * RBF(length_scale=67.0)

# seasonal component
k2 = 2.4**2 * RBF(length_scale=90.0) * \
        ExpSineSquared(length_scale=1.3, periodicity=1.0)

# medium term irregularity
k3 = 0.66**2 * RationalQuadratic(length_scale=1.2, alpha=0.78)

# noise terms
k4 = 0.18**2 * RBF(length_scale=0.134) + \
        WhiteKernel(noise_level=0.19**2)
kernel_gpml = k1 + k2 + k3 + k4

gp = GaussianProcessRegressor(kernel=kernel_gpml, alpha=0,
                              optimizer=None, normalize_y=True)
gp.fit(X, y)

print("GPML kernel: %s" % gp.kernel_)
print("Log-marginal-likelihood: %.3f"
      % gp.log_marginal_likelihood(gp.kernel_.theta))

# Kernel with optimized parameters

# long term smooth rising trend
k1 = 50.0**2 * RBF(length_scale=50.0)

# seasonal component
k2 = 2.0**2 * RBF(length_scale=100.0) * \
        ExpSineSquared(length_scale=1.0, periodicity=1.0,
                       periodicity_bounds="fixed")

# medium term irregularities
k3 = 0.5**2 * RationalQuadratic(length_scale=1.0, alpha=1.0)

# noise terms
k4 = 0.1**2 * RBF(length_scale=0.1) + \
        WhiteKernel(noise_level=0.1**2,
                   noise_level_bounds=(1e-3, np.inf))
kernel = k1 + k2 + k3 + k4

gp = GaussianProcessRegressor(kernel=kernel, alpha=0,
                              normalize_y=True)
gp.fit(X, y)

print("\nLearned kernel: %s" % gp.kernel_)
print("Log-marginal-likelihood: %.3f"
      % gp.log_marginal_likelihood(gp.kernel_.theta))

X_ = np.linspace(X.min(), X.max() + 32, 1000)[:, np.newaxis]
y_pred, y_std = gp.predict(X_, return_std=True)

fig = plt.figure(figsize=(9, 6))
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['font.sans-serif'] = "Helvetica"
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['font.size'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

ax = fig.add_subplot(111)
ax.plot(df.decimal_date, df.interpolated, color="royalblue", ls='-')

current_yr = 2018
YY = y_pred[X_[:, 0] > current_yr]
XX = X_[X_[:, 0] > current_yr]
ax.plot(XX, YY, color="salmon", ls='-')
ax.fill_between(X_[:, 0], y_pred - y_std, y_pred + y_std,
                 alpha=0.5, color='k')

ax.set_xlabel("Year")
ax.set_ylabel(u"CO$_2$ concentration (\u03BCmol mol$^{-1}$)")
plt.show()
