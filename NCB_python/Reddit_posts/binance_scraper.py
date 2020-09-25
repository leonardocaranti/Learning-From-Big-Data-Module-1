from binance.client import Client
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dts


# Binance API and account details
api_label = ''
api_key = ''
api_secret = ''
email = ''
password = ''

client = Client(api_key, api_secret)

def date_from_timestamp(ts):
    dot_idx = str(ts).find('.')
    if dot_idx != -1 or len(str(ts)) > 10:
        new_ts = float(str(ts)[:10])
    else:
        new_ts = float(ts)
    return datetime.datetime.fromtimestamp(new_ts)

def date_to_ordinal(date_str):
    date_str = str(date_str)
    day = int(date_str[8:10])
    month = int(date_str[5:7])
    year = int(date_str[0:4])
    dt_date = datetime.date(year, month, day)
    if len(date_str) > 10:
        hours = int(date_str[11:13])
        mins = int(date_str[14:16])
        secs = int(date_str[17:19])
        dt_date = datetime.datetime(year, month, day, hours, mins, secs)

    return dts.date2num(dt_date)

def date_to_timestamp(date_str):
    day = int(date_str[8:10])
    month = int(date_str[5:7])
    year = int(date_str[0:4])

    ts = datetime.datetime.timestamp(datetime.datetime(year, month, day))
    return ts

def plot_currency(curr_str, start_date, end_date = None, interval = '12h'):

    curr_str = str(curr_str).upper()

    # valid intervals - 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
    if end_date == None:
        bars = client.get_historical_klines(curr_str + 'USDT', interval, str(date_to_timestamp(start_date)), limit=1000)
    else: 
        bars = client.get_historical_klines(curr_str + 'USDT', interval, str(date_to_timestamp(start_date)), str(date_to_timestamp(end_date)), limit=1000)
    bars = np.array(bars)


    # Plotting
    dates = np.array([date_to_ordinal(date_from_timestamp(i)) for i in bars[:, 6]])
    close_vals = bars[:, 4].astype('float64')
    plt.title(f'{curr_str}-USDT {interval} Close Value Rate Since {start_date}')
    plt.plot_date(dates, close_vals, ls='-', color = 'k', marker='.')
    plt.xlabel('Date')
    plt.ylabel(f' {curr_str}-USDT rate ($)')
    plt.minorticks_on()
    plt.grid(which='major', linestyle='dotted', color='k', alpha=0.8)
    plt.grid(which='minor', linestyle='dotted', color='k', alpha=0.3)
    plt.show()
