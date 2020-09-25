import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date as date_fn
from os import listdir
from os.path import isfile, join
from binance_scraper import plot_currency
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression


folder = 'Daily_stats/'
files_list = [f for f in listdir(folder) if isfile(join(folder, f))]


def count_total_comments():
    total = 0
    for i, file_name in enumerate(files_list):
        df = pd.read_csv(folder + file_name, sep=',', header=0, index_col = 0, engine='python',encoding='utf-8-sig')
        row = df.loc[df.index == 'total']
        daily_tot = row['# comments'].to_numpy()[0]
        total += daily_tot
    return total


def average_param(symbol, param):
    param_list = []
    for i, file_name in enumerate(files_list):
        df = pd.read_csv(folder + file_name, sep=',', header=0, index_col = 0, engine='python',encoding='utf-8-sig')
        row = df.loc[df.index == symbol]
        daily_param = row[param].to_numpy()
        if len(daily_param) == 0:
            param_list.append(np.array([np.NaN]))
        else:
            param_list.append(daily_param)
    return np.nanmean(np.array(param_list))


def date_to_ordinal(date_str):
    day = int(date_str[0:2])
    month = int(date_str[3:5])
    year = int(date_str[6:10])

    ordinal = date_fn(year, month, day).toordinal()
    return ordinal


def plot_param_over_time(row_name, parameter): 
    dates, y = np.zeros((len(files_list))), np.zeros((len(files_list)))
    degree = 3 
    polyreg = make_pipeline(PolynomialFeatures(degree),LinearRegression())

    for i, file_name in enumerate(files_list):
        df = pd.read_csv(folder + file_name, sep=',', header=0, index_col = 0, engine='python',encoding='utf-8-sig')
        date = file_name[:-4]

        row = df.loc[df.index == row_name]

        new_val = row[parameter].to_numpy()
        if new_val.size == 0:
            if parameter[-5:] == 'score': 
                y[i] = 0.2
            else:
                y[i] = 0.5
        else:
            y[i] = new_val
        
        dates[i] = date_to_ordinal(date)

    polyreg.fit(dates.reshape(-1,1),y.reshape(-1,1))

    y, dates = y[dates.argsort()], dates[dates.argsort()]
    plt.title(row_name + ' ' + parameter + ' distribution')
    plt.xlabel('date')
    plt.ylabel(parameter)
    plt.minorticks_on()
    plt.grid(which='major', linestyle='dotted', color='k', alpha=0.8)
    plt.grid(which='minor', linestyle='dotted', color='k', alpha=0.3)
    plt.plot_date(dates, polyreg.predict(dates.reshape(-1,1)), ls='-', marker=None, color='red', alpha = 0.5)
    plt.plot_date(dates, y, ls='-', marker='.', color='black')


def print_avgs():

    print('Average positiveness for btc:', average_param('btc', 'mean P(pos)').round(5))
    print('Average positiveness for eth:', average_param('eth', 'mean P(pos)').round(5))
    print('Average positiveness for link:', average_param('link', 'mean P(pos)').round(5))
    print('Average positiveness for xrp:', average_param('xrp', 'mean P(pos)').round(5))
    print('Average positiveness for ada:', average_param('ada', 'mean P(pos)').round(5))
    print('Average positiveness for bnb:', average_param('bnb', 'mean P(pos)').round(5))
    print('Average positiveness for ltc:', average_param('ltc', 'mean P(pos)').round(5))
    print('Average positiveness for dot:', average_param('dot', 'mean P(pos)').round(5))
    print('Average positiveness for vet:', average_param('vet', 'mean P(pos)').round(5), '\n')
    print('Average uptrend for btc:', average_param('btc', 'mean P(up)').round(5))
    print('Average uptrend for eth:', average_param('eth', 'mean P(up)').round(5))
    print('Average uptrend for link:', average_param('link', 'mean P(up)').round(5))
    print('Average uptrend for xrp:', average_param('xrp', 'mean P(up)').round(5))
    print('Average uptrend for ada:', average_param('ada', 'mean P(up)').round(5))
    print('Average uptrend for bnb:', average_param('bnb', 'mean P(up)').round(5))
    print('Average uptrend for ltc:', average_param('ltc', 'mean P(up)').round(5))
    print('Average uptrend for dot:', average_param('dot', 'mean P(up)').round(5))
    print('Average uptrend for vet:', average_param('vet', 'mean P(up)').round(5), '\n')


# print_avgs()

coin_name = 'bnb'

plt.subplot(211)
plot_param_over_time(coin_name, 'mean P(up)')
plt.subplot(212)
plot_param_over_time(coin_name, '# comments')

plt.figure()
plt.subplot(211)
plot_param_over_time(coin_name, 'mean P(pos)')
plt.subplot(212)
plot_param_over_time(coin_name, 'mean P(neg)')

plt.figure()
plot_currency(coin_name, '2020-07-01')
plt.show() 

