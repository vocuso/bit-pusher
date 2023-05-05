import os
import math
import sqlite3
import requests
import datetime
import configparser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from matplotlib.gridspec import GridSpec
from binance.client import Client
from datetime import date

database_path = "database/crypto.db"
config = configparser.ConfigParser()
config.read('config.ini')

# 从CSV文件初始化数据库
def init_database_from_file(price_csv_file, fgi_csv_file, table_name):
    cali_price_csv_file = price_csv_file.replace(".csv", "_cali.csv")
    if os.path.exists(cali_price_csv_file):
        os.remove(cali_price_csv_file)
    if not os.path.exists(price_csv_file):
        return

    src_file = open(price_csv_file, "r")
    dst_file = open(cali_price_csv_file, "w")

    line = src_file.readline()
    while line:
        if line.find(",") == -1:
            line = src_file.readline()
            continue

        line_str_list = line.strip().split(',')
        if not line_str_list[-1].isnumeric():
            dst_file.write(line)
            line = src_file.readline()
            continue

        dst_file.write(','.join(line_str_list) + "\n")
        line = src_file.readline()

    src_file.close()
    dst_file.close()

    conn = sqlite3.connect(database_path)
    price_df = pd.read_csv(cali_price_csv_file)
    fgi_df = pd.read_csv(fgi_csv_file)

    price_df.to_sql(table_name, conn, if_exists='replace', index=False)
    fgi_df.to_sql("fgi", conn, if_exists='replace', index=False)
    os.remove(cali_price_csv_file)
    conn.close()

# 从文件中补全数据到数据库
def complete_data_to_database(src_file_path):
    src_file = open(src_file_path, "r")

    # 连接到SQLite数据库
    conn = sqlite3.connect(database_path)
    c = conn.cursor()

    line = src_file.readline()
    while line:
        if line.find(",") == -1 or (not line.strip().split(',')[-1].isnumeric()):
            line = src_file.readline()
            continue

        line_str_list = line.strip().split(',')
        line = src_file.readline()

        timestamp = float(line_str_list[0])
        current_date = datetime.datetime.fromtimestamp(
            timestamp).strftime('%Y-%m-%d')
        symbol = line_str_list[2]
        open_price = float(line_str_list[3])
        high_price = float(line_str_list[4])
        low_price = float(line_str_list[5])
        close_price = float(line_str_list[6])
        btc_volume = float(line_str_list[7])
        usdt_volume = float(line_str_list[8])
        trade_count = float(line_str_list[9])

        # 检查日期是否存在于表中
        c.execute("SELECT COUNT(*) FROM btcusdt WHERE Date=?", (current_date,))
        data_exists = c.fetchone()[0]

        # 如果日期不存在，则将数据插入表中
        if not data_exists:
            c.execute("INSERT INTO btcusdt VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (timestamp, current_date, symbol,
                                                                                    open_price, high_price, low_price, close_price, btc_volume, usdt_volume, trade_count))

    # 提交更改并关闭连接
    conn.commit()
    conn.close()

# 更新最新的数据到数据库
def update_data_to_database():
    api_key = config.get('Binance', 'api_key')
    api_secret = config.get('Binance', 'api_secret')
    client = Client(api_key, api_secret)

    symbol = 'BTCUSDT'
    start_date = '2023-01-01'  # 指定起始日期
    end_date = str(date.today())  # 指定结束日期
    interval = Client.KLINE_INTERVAL_1DAY  # 指定时间间隔为1天

    start_timestamp = int(datetime.datetime.strptime(
        start_date, '%Y-%m-%d').timestamp() * 1000)
    end_timestamp = int(datetime.datetime.strptime(
        end_date, '%Y-%m-%d').timestamp() * 1000)

    klines = client.get_historical_klines(
        symbol, interval, start_timestamp, end_timestamp)

    # 连接到SQLite数据库
    conn = sqlite3.connect(database_path)
    c = conn.cursor()

    for kline in klines:
        current_date = datetime.datetime.fromtimestamp(
            kline[0]/1000).strftime('%Y-%m-%d')
        symbol = "BTCUSDT"
        open_price = float(kline[1])
        high_price = float(kline[2])
        low_price = float(kline[3])
        close_price = float(kline[4])
        btc_volume = float(kline[5])
        usdt_volume = float(kline[7])
        trade_count = float(kline[8])

        # 检查日期是否存在于表中
        c.execute("SELECT COUNT(*) FROM btcusdt WHERE Date=?", (current_date,))
        data_exists = c.fetchone()[0]

        # 如果日期不存在，则将数据插入表中
        if not data_exists:
            c.execute("INSERT INTO btcusdt VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (kline[0], current_date, symbol,
                                                                                    open_price, high_price, low_price, close_price, btc_volume, usdt_volume, trade_count))

    # 构建API请求URL
    url = f'https://api.alternative.me/fng/?limit=1000&date_format=cn&start_date={start_date}&end_date={end_date}'

    # 发送API请求并处理响应
    response = requests.get(url)
    data = response.json()

    for record in data['data']:
        if record['timestamp'] == end_date:
            continue

        # 检查日期是否存在于表中
        c.execute("SELECT COUNT(*) FROM fgi WHERE Date=?",
                  (record['timestamp'],))
        data_exists = c.fetchone()[0]

        # 如果日期不存在，则将数据插入表中
        if not data_exists:
            c.execute("INSERT INTO fgi VALUES (?, ?, ?)", (record['timestamp'], int(
                record['value']), record['value_classification']))

    # 提交更改并关闭连接
    conn.commit()
    conn.close()

# 获取最新数据日期
def get_latest_date():
    now = datetime.datetime.now()
    latest_date = now if now.time() >= datetime.time(hour=8) else (now - datetime.timedelta(days=1))
    print(f'latest_date : {latest_date}')
    return latest_date.strftime('%Y-%m-%d')

# 获取 FGI 指数
def get_fgi_index(datetime):
    conn = sqlite3.connect(database_path)
    condition = "WHERE Date = '%s'" % (datetime)
    fgi_df = pd.read_sql_query("SELECT * FROM fgi " + condition, conn)
    fgi_df = fgi_df.sort_values('Date', ascending=True)
    conn.close()

    if len(fgi_df.columns) == 0 : return None

    fgi_value = fgi_df['Value'].tolist()[0]
    fgi_class = fgi_df['Classification'].tolist()[0]
    return {'value' : fgi_value, 'class' : fgi_class}

# 把指定时间范围的数据加载到 dataframe
def read_data_to_dataframe(table_name, start_time, end_time, read_fgi_data=False):
    condition = " WHERE Date >= '%s' AND Date <= '%s'" % (start_time, end_time)
    conn = sqlite3.connect(database_path)
    df = pd.read_sql_query("SELECT * FROM " + table_name + condition, conn)
    df = df.sort_values('Date', ascending=True)

    if read_fgi_data:
        fgi_df = pd.read_sql_query("SELECT * FROM fgi " + condition, conn)
        fgi_df = fgi_df.sort_values('Date', ascending=True)
        df['FGI'] = fgi_df['Value'].tolist()

    conn.close()
    return df

# 根据 dataframe 数据画价格曲线、多均线曲线和FGI指数曲线
def show_data_to_plot(dataframe, indicator, ma_list=[], profit_rate_list=[], show_fgi_plot=False):
    x = dataframe['Date'].tolist()
    y = dataframe[indicator].tolist()
    if show_fgi_plot:
        y2 = dataframe['FGI'].tolist()

    xticks = list(range(0, len(x), max([len(x) // 40, 1])))
    xlabels = [x[i] for i in xticks]

    figure = plt.figure(figsize=(38, 8))
    gs = GridSpec(1, 1)
    ax = plt.subplot(gs[0, 0])
    ax.plot(x, y, label='Price')

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=70)

    ax.set_xlabel("Time (d)")
    ax.set_ylabel("Price (USDT)")
    ax.legend(loc=2)

    for ma in ma_list:
        ma_dataframe = dataframe.rolling(window=ma)[[indicator]].mean()
        ax.plot(x, ma_dataframe[indicator].tolist(), label='MA'+str(ma))
        # print(f'price date num : {(y)}')
        # print(f'ma{ma} date num : {(ma_dataframe[indicator].tolist())}')

    if len(profit_rate_list) == 0 and show_fgi_plot:
        ax2 = ax.twinx()
        ax2.plot(x, y2, 'r', label='FGI')
        ax2.set_ylabel("Fear&Greed Index")
        ax2.legend(loc=1)

    if len(profit_rate_list) == len(x):
        ax2 = ax.twinx()
        ax2.plot(x, profit_rate_list, 'r', label='Profit Rate')
        ax2.set_ylabel("Profit Rate (%)")
        ax2.legend(loc=1)

        ax2.set_xticks(xticks)
        ax2.set_xticklabels(xlabels, rotation=70)

    plt.savefig('plot.png')

# 统计均线变化率曲线
def show_ma_curve_gradient_to_plot(dataframe, indicator, ma, window_size):
    date_list = dataframe['Date'].tolist()
    price_list = dataframe[indicator].tolist()
    ma_price_list = (dataframe.rolling(window=ma)[
                     ['Close']].mean())['Close'].tolist()

    ma_gradient_list = list()
    for i in range(len(date_list)):
        if math.isnan(ma_price_list[i]) or (i > window_size - 1 and
           math.isnan(ma_price_list[i - window_size + 1])):
            ma_gradient_list.append(0)
            continue
        diff_value = ma_price_list[i] - ma_price_list[i - window_size + 1]
        diff_rate = diff_value / ma_price_list[i - window_size + 1]
        ma_gradient_list.append(diff_rate)

    xticks = list(range(0, len(date_list), max([len(date_list) // 40, 1])))
    xlabels = [date_list[i] for i in xticks]

    figure = plt.figure(figsize=(38, 8))
    gs = GridSpec(1, 1)
    ax = plt.subplot(gs[0, 0])
    ax.plot(date_list, price_list, label='price')

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=70)

    ma_dataframe = dataframe.rolling(window=ma)[[indicator]].mean()
    ax.plot(date_list, ma_dataframe[indicator].tolist(), label='ma'+str(ma))

    ax.set_xlabel("Time (d)")
    ax.set_ylabel("Price (USDT)")
    ax.legend(loc=2)

    ax2 = ax.twinx()
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xlabels, rotation=70)
    ax2.plot(date_list, ma_gradient_list, 'r', label='ma_diff_rate')
    ax2.set_ylabel("MA Diff Rate (%)")
    ax2.legend(loc=1)

# 统计并画出价格均线差值曲线及其分布
def show_data_with_diff_rate_to_plot(dataframe, indicator, near_ma, mid_ma, upper_diff=False):
    date_list = dataframe['Date'].tolist()
    price_list = dataframe[indicator].tolist()
    near_ma_price_list = (dataframe.rolling(window=near_ma)[
                          ['Close']].mean())['Close'].tolist()
    mid_ma_price_list = (dataframe.rolling(window=mid_ma)[
        ['Close']].mean())['Close'].tolist()

    # 计算价格与均线差值百分比
    diff_rate_list = list()
    frequency_list = [0] * 6
    x = [10, 20, 30, 40, 50, 100]
    for i in range(len(date_list)):
        diff_value = min(near_ma_price_list[i] - mid_ma_price_list[i], 0.0)
        if upper_diff:
            diff_value = max(near_ma_price_list[i] - mid_ma_price_list[i], 0.0)
        # diff_value = near_ma_price_list[i] - mid_ma_price_list[i]
        diff_rate = diff_value / mid_ma_price_list[i] * 100
        diff_rate_list.append(diff_rate)

        if ((not upper_diff) and near_ma_price_list[i] - mid_ma_price_list[i] < 0.0) or \
           (upper_diff and near_ma_price_list[i] - mid_ma_price_list[i] > 0.0):
            # print(f'date : {date_list[i]}, price : {price_list[i]}, diff_rate : {diff_rate_list[i]}%')
            for k, x_area in enumerate(x):
                if abs(diff_rate) >= x_area:
                    continue
                frequency_list[k] += 1
                break

    xticks = list(range(0, len(date_list), max([len(date_list) // 40, 1])))
    xlabels = [date_list[i] for i in xticks]

    figure = plt.figure(figsize=(38, 8))
    gs = GridSpec(1, 1)
    ax = plt.subplot(gs[0, 0])
    ax.plot(date_list, price_list, label='price')

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=70)

    near_ma_dataframe = dataframe.rolling(window=near_ma)[[indicator]].mean()
    mid_ma_dataframe = dataframe.rolling(window=mid_ma)[[indicator]].mean()
    ax.plot(date_list, near_ma_dataframe[indicator].tolist(
    ), label='ma'+str(near_ma))
    ax.plot(date_list, mid_ma_dataframe[indicator].tolist(
    ), label='ma'+str(mid_ma))

    ax.set_xlabel("Time (d)")
    ax.set_ylabel("Price (USDT)")
    ax.legend(loc=2)

    ax2 = ax.twinx()
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xlabels, rotation=70)
    ax2.plot(date_list, diff_rate_list, 'r', label='diff_rate')
    ax2.set_ylabel("Diff Rate (%)")
    ax2.legend(loc=1)

    # 价格均线差值百分比分布
    xticks = [10, 20, 30, 40, 50, 60]
    figure = plt.figure(figsize=(38, 8))
    plt.bar(x, frequency_list)
    plt.xticks(xticks)
    plt.xlabel("Down Diff Area (%)")
    plt.ylabel("Frequency (Times)")
    plt.show()

# 统计并画出 FGI 分布
def show_fgi_distribution_to_plot():
    conn = sqlite3.connect(database_path)
    fgi_df = pd.read_sql_query("SELECT * FROM fgi", conn)
    fgi_df = fgi_df.sort_values('Date', ascending=True)
    date_list = fgi_df['Date'].tolist()

    condition = " WHERE Date >= '%s' AND Date <= '%s'" % (
        date_list[0], date_list[-1])
    conn = sqlite3.connect(database_path)
    price_df = pd.read_sql_query("SELECT * FROM btcusdt " + condition, conn)
    price_df = price_df.sort_values('Date', ascending=True)
    conn.close()

    price_list = price_df['Close'].tolist()
    ma_price_list = (price_df.rolling(window=120)[
                     ['Close']].mean())['Close'].tolist()
    lower_area_list = list()
    down_forward_i = 1
    for i in range(1, len(date_list)):

        # 价格上穿均线
        if price_list[i-1] <= ma_price_list[i-1] and price_list[i] > ma_price_list[i]:
            lower_area_list.append((down_forward_i, i - 1))

        # 价格下穿均线
        if price_list[i-1] >= ma_price_list[i-1] and price_list[i] < ma_price_list[i]:
            down_forward_i = i

    value_list = fgi_df['Value'].tolist()
    print(f'value_list_len : {len(value_list)}')

    x = list(range(1, 101))
    frequency_list = [0] * 100
    lower_area_value_list = list()
    upper_area_value_list = list()

    for date_i, value in enumerate(value_list):
        for index in x:
            if value != index:
                continue
            frequency_list[index] += 1
            break

        in_lower_area = False
        for lower_area in lower_area_list:
            if date_i >= lower_area[0] and date_i <= lower_area[1]:
                lower_area_value_list.append(value)
                in_lower_area = True
                # if value <= 20:
                #     print(date_list[date_i], price_list[date_i])

        if not in_lower_area:
            upper_area_value_list.append(value)
            # if value >= 80:
            #     print(date_list[date_i], price_list[date_i])

    print(f'lower_area_mean_value : {sum(lower_area_value_list) / len(lower_area_value_list)}')
    print(f'upper_area_mean_value : {sum(upper_area_value_list) / len(upper_area_value_list)}')
    print(f'max_value_in_lower_area : {max(lower_area_value_list)}')
    print(f'min_value_in_upper_area : {min(upper_area_value_list)}')

    xticks = list(range(0, len(x) + 1, 2))
    figure = plt.figure(figsize=(38, 8))
    plt.bar(x, frequency_list)
    plt.xticks(xticks)
    plt.xlabel("Index Value (N/A)")
    plt.ylabel("Frequency (Times)")
    plt.show()

# 获取指定交易对实时价格
def get_current_price(sym):
    api_key = config.get('Binance', 'api_key')
    api_secret = config.get('Binance', 'api_secret')
    client = Client(api_key, api_secret)
    current_price = client.get_symbol_ticker(symbol=sym)

    return current_price
