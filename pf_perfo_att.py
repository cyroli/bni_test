"""
Created on Dec 13, 2022

Technical test for Quantitative Analyst role at BNI - CIO office

@author: OLIVIER CYR-CHOINIÈRE
"""


#################################################################################
### Import relevant python libraries

import numpy as np
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt

#################################################################################

# Define names of paths and files
data_path = ''
file_name = 'Technical Test - Portfolio Attribution.xlsm'
bench_sheet_name = "Benchmark Weights"
saa_sheet_name = "SAA Weights"
manager_sheet_name = "Manager Weights"
returns_sheet_name = "Returns"

#################################################################################
# Load data from xlsm file sheets
# Use multiple lines headers as multi-index

bench_w = pd.read_excel(data_path + file_name, sheet_name=bench_sheet_name, header=[0, 1], index_col=[0])
saa_w = pd.read_excel(data_path + file_name, sheet_name=saa_sheet_name, header=[0, 1], index_col=[0])
manager_w = pd.read_excel(data_path + file_name, sheet_name=manager_sheet_name, header=[0, 1, 2], index_col=[0])
rets = pd.read_excel(data_path + file_name, sheet_name=returns_sheet_name, header=[0, 1], index_col=[0])


# Check to see number of missing values per columns in returns
print('Number of NA values:\n', rets.isnull().sum())


##################################################################################
# Functions

def PF_rets(w, rets, name):
    # Function to do compute returns of portfolio given assets' weights and returns
    #  INPUTS
    #   w         : [matrix]  weights of each asset class at different times in given portfolio
    #   rets      : [matrix]  returns of each asset class at different times
    #   name      : [string]  name of given portfolio
    #  OUTPUTS
    #   rets_pf   : [vector]  returns of portfolio at different times

    # Extract dates of allocation changes
    dates_pf_change = w.index

    # Initialize dataframe of saa returns
    rets_pf = pd.DataFrame()

    # Loop on dates of allocation changes
    for i in range(0, len(dates_pf_change)):
        # Select weights between dates of allocation changes
        w_i = w.iloc[i]

        # Select subset of returns between dates of allocation changes
        # Weight allocation change will be effective on next day return; exclude (include) lower (upper) bound
        if i == 0:  # Last recent date
            rets_i = rets.loc[(rets.index > dates_pf_change[i])]
        else:
            rets_i = rets.loc[(rets.index > dates_pf_change[i]) &
                              (rets.index <= dates_pf_change[i-1])]

        # Multiply matrix of returns with matrix of corresponding weights for given dates
        # rets_pf_i = rets_i @ w_i.values.T

        # Alternate way to make dot product to ignore NA values, so they don't propagate
        # Product of assets' returns and weights is taken for each asset and summed over each row, by ignoring NAs
        rets_pf_i = (rets_i * w_i.values.T).sum(axis=1, skipna=True)

        # Add subset of returns to full dataframe of returns
        rets_pf = pd.concat([rets_pf, rets_pf_i])

    # Rename returns columns
    rets_pf.columns = [name + ' portfolio returns']

    return rets_pf

############################


def PF_cum_rets(rets, name):
    # Function to compute cumulative returns of portfolio daily returns
    #  INPUTS
    #   rets      : [vector]  returns of portfolio at different times
    #   name      : [string]  name of given portfolio
    #  OUTPUTS
    #   cum_rets  : [vector]  cumulative returns of portfolio as function of time

    # Inverse order of dataframe to sum chronologically,
    # from older (last row) to recent (first row) date
    # and re-inverse to get cumulative time-series in same order as before

    # Cumulative returns is the product of daily gross return; (1 + r_i)
    cum_rets = (np.cumprod(1 + rets[::-1]) - 1)[::-1]

    # Add 1 to series, so that initial investment is normalized to 1
    cum_rets = cum_rets + 1

    # Rename cumulative returns columns
    cum_rets.columns = [name + ' portfolio cumulative returns']

    return cum_rets


############################

def PF_index(rets, norm_date, factor, name):
    # Function to construct portfolio index from daily returns of portfolio
    #  INPUTS
    #   rets      : [vector]  returns of portfolio at different times
    #   norm_date : [string]  date at which index is normalized
    #   factor    : [scalar]  factor to scale the index (1, 100, 1000)
    #   name      : [string]  name of given portfolio
    #  OUTPUTS
    #   index_pf  : [vector]  index value of portfolio as function of time

    # Function to compute cumulative returns
    cum_rets = PF_cum_rets(rets, name)

    # Normalize cumulative returns by its value at a given date
    # and multiply by factor 100
    index_pf = cum_rets / cum_rets.loc[norm_date] * 100

    # Rename index columns
    index_pf.columns = [name + ' portfolio index']

    return index_pf

##################################################################################


######################################
### Section 1. Portfolio Return Series
######################################
"""
"Construct the following portfolios' daily return series:
1. Benchmark portfolio (daily rebalanced)
2. SAA portfolio (daily rebalanced)
3. Manager portfolio (daily rebalanced)
(BONUS) 4. Manager portfolio (rebalanced on target dates only, drift between)"
"""
# Assuming that PF is rebalanced daily at end of day --> always keep the initial weights
# Since initial weights are kept, we simply need to use this same set of weights throughout the time series of returns
# PortFolio returns are extract from multiplying
# the weights of each class with the return from the given class of the index

##############################
# 1.1 Benchmark PF returns

# Define sub-section of returns as returns of Index
rets_index = rets[rets.columns[0][0]]

# Last asset class of Index is not used in Benchmark
rets_index_bench = rets_index.iloc[:, :-1]

bench_rets = PF_rets(bench_w, rets_index_bench, 'Benchmark')


##############################
# 1.2 SAA PF returns

saa_rets = PF_rets(saa_w, rets_index, 'SAA')


##############################
# 1.3 Manager PF returns

# Define sub-section of returns as returns of funds
rets_funds = rets[rets.columns[-1][0]]

# Need to get weights of fund according to strategic asset allocation (saa)
# Need to multiply funds weights with allocation weights to get overall weights

dates_pf_change = saa_w.index

# Initialize dataframe of manager adjusted weights
manager_w_adj = pd.DataFrame()

# Loop on dates of allocation changes
for i in range(0, len(dates_pf_change)):
    # Select saa weights between dates of allocation changes
    saa_w_i = saa_w.iloc[i]

    # Select subset of manager weights between dates of allocation changes
    if i == 0:  # Last recent date
        w_manager_i = manager_w.loc[(manager_w.index > dates_pf_change[i])]
    else:
        w_manager_i = manager_w.loc[(manager_w.index >= dates_pf_change[i]) &
                                    (manager_w.index < dates_pf_change[i-1])]

    # Multiply vector of saa weights at given dates with vector of manager (funds) weights for given dates
    w_manager_adj_i = pd.DataFrame()
    for j in saa_w.columns:
        w_manager_adj_i_j = w_manager_i[j] * saa_w_i[j]
        w_manager_adj_i = pd.concat([w_manager_adj_i, w_manager_adj_i_j], axis=1)

    manager_w_adj = pd.concat([manager_w_adj, w_manager_adj_i])


# Compute returns of manager portfolio
manager_rets = PF_rets(manager_w_adj, rets_funds, 'Manager')

##############################
# 1.4 Manager PF returns, rebalanced on target dates only

"""
# Compute cumulative returns of funds
cum_rets = (np.cumprod(1 + rets_funds[::-1]) - 1)[::-1]

dates_pf_change = manager_w.index

# Initialize dataframe of manager drifted weights
manager_w_drift = pd.DataFrame()

# Loop on dates of manager allocation changes
for i in range(0, len(dates_pf_change)):
    # Select manager weights between dates of allocation changes
    w_i = manager_w_adj.iloc[i]

    # Select subset of cumulative returns between dates of allocation changes
    cum_rets_i = cum_rets.loc[(cum_rets.index > dates_pf_change[i]) &
                              (cum_rets.index <= dates_pf_change[i-1])]

    # Compute weighted fund values
    cum_rets_ass_pf = cum_rets_i.multiply(w_i.values)

    # Compute sum of weigthed fund values (total is PF)
    pf_val = cum_rets_ass_pf.sum(axis=1)

    # Get drifted weights of assets by normalizing funds values by total PF
    w_i_drift = (cum_rets_ass_pf.T/pf_val).T

    manager_w_drift = pd.concat([manager_w_drift, w_i_drift])


# Compute returns of manager drifted portfolio
manager_rets_drift = PF_rets(manager_w_drift, rets_funds, 'Manager target dates rebalancing'')

"""

# Combine daily returns of portfolios
pf_rets = pd.concat([bench_rets, saa_rets, manager_rets], axis=1)

# Save dataframes to csv
pf_rets.to_csv('portfolio_returns.csv')#, index=False)

######################################
### Section 2. Portfolio Index Series
######################################
"""
Construct an index, normalized at 100, for each of the portfolios above. 
The start date for the normalization should be an argument which can be easily changed. 
"""
# Index construction from the portfolio daily returns is
# constructed by taking the cumulative returns and
# normalize this cumulative series by its value at a given date and multiply it by 100

normalization_date = '2020-03-06'
scaling_factor = 100

##############################
# 2.1 Benchmark PF Index

bench_index = PF_index(bench_rets, normalization_date, scaling_factor, 'Benchmark')

##############################
# 2.2 SAA PF Index

saa_index = PF_index(saa_rets, normalization_date, scaling_factor, 'SAA')

##############################
# 2.3 Manager PF Index

manager_index = PF_index(manager_rets, normalization_date, scaling_factor, 'Manager')


##############################
# 2.4 Manager PF returns, rebalanced on target dates only

"""
manager_drift_index = PF_index(manager_rets_drift, normalization_date, scaling_factor, 'Manager target dates rebalancing')
"""

# Combine index values of portfolios
pf_index = pd.concat([bench_index, saa_index, manager_index], axis=1)

# Save dataframes to csv
pf_index.to_csv('portfolio_indices.csv')


#########################################
### Section 3. Cumulative Outperformance
#########################################
"""
Construct a timeseries of the cumulative out-/under-performance 
between each successive pair of portfolios (1 vs 2, 2 vs 3, 3 vs 4). 
"""
# Out-/under-performance of portfolios is simply the difference in
# the cumulative returns between given portfolios

# Extract cumulative returns of each portfolio
bench_cum_rets = PF_cum_rets(bench_rets, 'Benchmark')
saa_cum_rets = PF_cum_rets(saa_rets, 'SAA')
manager_cum_rets = PF_cum_rets(manager_rets, 'Manager')
#manager_drift_cum_rets = PF_cum_rets(manager_drift_rets, 'Manager target dates rebalancing')



##############################
# 3.1 Out-/under-performance of SAA vs Benchmark

# Compute difference between cumulative returns
perfo_saa_bench = saa_cum_rets - bench_cum_rets.values

# Rename columns
perfo_saa_bench.columns = ['Out-Performance SAA vs Benchmark']


##############################
# 3.2 Out-/under-performance of Manager vs SAA

# Compute difference between cumulative returns
perfo_manager_saa = manager_cum_rets - saa_cum_rets.values

# Rename columns
perfo_manager_saa.columns = ['Out-Performance Manager vs SAA']


##############################
# 3.3 Out-/under-performance of Manager-target-dates-rebalanced vs Manager-daily-rebalanced

"""
# Compute difference between cumulative returns
perfo_manager_drift_bal = manager_drift_cum_rets - manager_cum_rets.values

# Rename columns
perfo_manager_drift_bal.columns = ['Out-Performance Manager Target balanced vs Daily balanced']
"""


# Combine out-performance values of portfolios
pf_perfo = pd.concat([perfo_saa_bench, perfo_manager_saa], axis=1)

# Save dataframes to csv
pf_perfo.to_csv('portfolio_performances.csv')


############################################################

# Plots

# Plot of PF daily returns
fig1, ax = plt.subplots(figsize=(8, 5), dpi=125)
plt.title('Portfolios daily returns')
ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Daily returns', fontsize=12)
pf_rets.plot(kind='line', lw=1.0, ax=ax)
ax.axhline(0, ls='-.', c='black', lw=0.5)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()


# Plot of PF index values
fig2, ax = plt.subplots(figsize=(8, 5), dpi=125)
plt.title('Portfolios daily returns')
ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Index value (normalized at 100)', fontsize=12)
pf_index.plot(kind='line', lw=1.0, ax=ax)
ax.axhline(100, ls='-.', c='black', lw=0.5)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()


# Plot of outperformance
fig3, ax = plt.subplots(figsize=(8, 5), dpi=125)
plt.title('Portfolios daily returns')
ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Portfolios Cumulative Outperformance', fontsize=12)
pf_perfo.plot(kind='line', lw=1.0, ax=ax)
ax.axhline(0, ls='-.', c='black', lw=0.5)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()
