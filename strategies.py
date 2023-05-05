
import os
import math
import sqlite3
import requests
import numpy as np
import pandas as pd
import utilities as utils
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from matplotlib.gridspec import GridSpec
from binance.client import Client
from datetime import datetime
from datetime import date

# 计算常规定投策略收益
def calculate_fixed_invest_strategy_profit(dataframe, fixed_invest_amount, use_open_price=True):
    day_price_list = [
        float(i) for i in dataframe['Open' if use_open_price else 'Close'].tolist()]
    day_num = len(day_price_list)

    invest_usdt_cumsum = [fixed_invest_amount *
                          i for i in range(1, day_num + 1)]
    invest_btc_cumsum = np.array(
        [(fixed_invest_amount / price) for price in day_price_list]).cumsum().tolist()
    asset_value_cumsum = [(invest_btc_cumsum[i] * day_price_list[i])
                          for i in range(day_num)]
    profit_rate = [((asset_value_cumsum[i] - invest_usdt_cumsum[i]) /
                   invest_usdt_cumsum[i] * 100) for i in range(day_num)]

    x = dataframe['Date'].tolist()
    xticks = list(range(0, len(x), max([len(x) // 40, 1])))
    xlabels = [x[i] for i in xticks]

    figure = plt.figure(figsize=(38, 8))
    gs = GridSpec(1, 1)
    ax = plt.subplot(gs[0, 0])
    ax.plot(x, invest_usdt_cumsum, label='invest_usdt_cumsum')
    ax.plot(x, asset_value_cumsum, label='asset_value_cumsum')

    ax2 = ax.twinx()
    ax2.plot(x, profit_rate, 'r', label='profit_rate')

    ax.set_xlabel("Time (d)")
    ax.set_ylabel("Value (USDT)")
    ax2.set_ylabel("Profit Rate (%)")

    ax.legend(loc=2)
    ax2.legend(loc=1)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=70)

    newest_usdt_cumsum = invest_usdt_cumsum[-1]
    newest_btc_cumsum = invest_btc_cumsum[-1]
    newest_asset_value = asset_value_cumsum[-1]
    newest_profit_rate = profit_rate[-1]

    max_usdt_cumsum = max(invest_usdt_cumsum)
    max_asset_value = max(asset_value_cumsum)
    max_profit_rate = max(profit_rate)

    max_usdt_cumsum_date = x[invest_usdt_cumsum.index(max_usdt_cumsum)]
    max_asset_value_date = x[asset_value_cumsum.index(max_asset_value)]
    max_profit_rate_date = x[profit_rate.index(max_profit_rate)]

    min_profit_rate = min(profit_rate)
    min_profit_rate_date = x[profit_rate.index(min_profit_rate)]

    print(f'newest_usdt_cumsum : {newest_usdt_cumsum}')
    print(f'newest_btc_cumsum : {newest_btc_cumsum}')
    print(f'newest_asset_value : {newest_asset_value}')
    print(f'newest_profit_rate : {newest_profit_rate}%')

    print(f'max_usdt_cumsum : {max_usdt_cumsum}, {max_usdt_cumsum_date}')
    print(f'max_asset_value : {max_asset_value}, {max_asset_value_date}')
    print(f'max_profit_rate : {max_profit_rate}%, {max_profit_rate_date}')
    print(f'min_profit_rate : {min_profit_rate}%, {min_profit_rate_date}')

# 计算FGI定投定卖策略收益
def calculate_fgi_strategy_profit(dataframe, buy_every_time=100, sell_every_time=100, buy_index=25, sell_index=63):
    usdt_cumsum = 0
    btc_cumsum = 0

    usdt_value_cumsum = 0
    max_usdt_invest = 0
    max_asset_value = 0
    max_asset_value_btc_num = 0
    max_asset_value_date = ""

    usdt_cumsum_list = list()
    rest_btc_num_list = list()
    rest_asset_value_list = list()
    profit_rate_list = list()

    for i in range(len(dataframe['Date'].tolist())):
        btc_price = dataframe['Close'].tolist()[i]
        btc_value_cumsum = btc_cumsum * btc_price
        if (dataframe['FGI'].tolist()[i] <= buy_index):
            if (usdt_value_cumsum >= buy_every_time):
                usdt_value_cumsum = usdt_value_cumsum - buy_every_time
            else:
                usdt_cumsum = usdt_cumsum + buy_every_time
            btc_cumsum = btc_cumsum + buy_every_time / btc_price
        if (dataframe['FGI'].tolist()[i] >= sell_index and btc_value_cumsum >= sell_every_time):
            usdt_value_cumsum = usdt_value_cumsum + sell_every_time
            btc_cumsum = btc_cumsum - sell_every_time / btc_price
        if btc_cumsum * btc_price + usdt_value_cumsum > max_asset_value:
            max_asset_value = btc_cumsum * btc_price + usdt_value_cumsum
            max_asset_value_date = dataframe['Date'].tolist()[i]
            max_asset_value_btc_num = btc_cumsum

        max_usdt_invest = usdt_cumsum if usdt_cumsum > max_usdt_invest else max_usdt_invest
        usdt_cumsum_list.append(usdt_cumsum)
        rest_btc_num_list.append(btc_cumsum)
        rest_asset_value_list.append(
            btc_cumsum * btc_price + usdt_value_cumsum)
        current_usdt_cumsum = usdt_cumsum_list[-1] if usdt_cumsum_list[-1] > 0 else 1
        profit_rate_list.append(
            (rest_asset_value_list[-1] - current_usdt_cumsum) / current_usdt_cumsum * 100)

    print(
        f'max_asset_value : {max_asset_value}, btc_num : {max_asset_value_btc_num}, {max_asset_value_date}')
    print(f'max_usdt_invest : {max_usdt_invest}')
    print(f'rest_btc_num : {btc_cumsum}')
    print(f'rest_usdt_cumsum : {usdt_value_cumsum}')
    print(f'rest_asset_value : {rest_asset_value_list[-1]}')

    x = dataframe['Date'].tolist()
    xticks = list(range(0, len(x), max([len(x) // 40, 1])))
    xlabels = [x[i] for i in xticks]

    figure = plt.figure(figsize=(38, 8))
    gs = GridSpec(1, 1)
    ax = plt.subplot(gs[0, 0])
    ax.plot(x, usdt_cumsum_list, label='invest_usdt_cumsum')
    ax.plot(x, rest_asset_value_list, label='rest_asset_value')

    ax2 = ax.twinx()
    ax2.plot(x, rest_btc_num_list, 'r', label='rest_btc_num')
    # ax2.plot(x, profit_rate_list, 'r', label='profit_rate')

    ax.set_xlabel("Time (d)")
    ax.set_ylabel("Value (USDT)")
    ax2.set_ylabel("Rest BTC")
    # ax2.set_ylabel("Profit Rate (%)")

    ax.legend(loc=2)
    ax2.legend(loc=1)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=70)

# 计算单均线策略收益
def calculate_single_ma_strategy_profit(dataframe, ma, init_invest_usdt):
    date_list = dataframe['Date'].tolist()
    price_list = dataframe['Close'].tolist()
    ma_price_list = (dataframe.rolling(window=int(ma))[
                     ['Close']].mean())['Close'].tolist()

    operation_record_list = list()
    accum_usdt_asset = init_invest_usdt
    accum_btc_asset = 0

    for i in range(1, len(date_list)):
        operation_record = dict()

        # 价格上穿均线
        if price_list[i-1] <= ma_price_list[i-1] and price_list[i] > ma_price_list[i]:
            accum_btc_asset = accum_usdt_asset / price_list[i]
            operation_record['date'] = date_list[i]
            operation_record['operation'] = 'BUY'
            operation_record['price'] = price_list[i]
            operation_record['asset'] = accum_usdt_asset
            operation_record_list.append(operation_record)

        # 价格下穿均线
        if price_list[i-1] >= ma_price_list[i-1] and price_list[i] < ma_price_list[i]:
            if accum_btc_asset == 0:
                continue
            accum_usdt_asset = accum_btc_asset * price_list[i]
            operation_record['date'] = date_list[i]
            operation_record['operation'] = 'SELL'
            operation_record['price'] = price_list[i]
            operation_record['asset'] = accum_usdt_asset
            operation_record_list.append(operation_record)

    print(f'total_rest_asset_usdt : {accum_usdt_asset}')
    for record in operation_record_list:
        print(record)

# 计算单均线定投策略收益
def calculate_ma_fixed_invest_strategy_profit(dataframe, ma, init_vest_usdt, min_fixed_invest_usdt, max_batch_days=200, batch_fixed_invest=False, fgi_enhancement=False, dr_enhancement=False):
    date_list = dataframe['Date'].tolist()
    price_list = dataframe['Close'].tolist()
    ma_price_list = (dataframe.rolling(window=int(ma))[
                     ['Close']].mean())['Close'].tolist()

    conn = sqlite3.connect(utils.database_path)
    fgi_df = pd.read_sql_query("SELECT * FROM fgi", conn)
    fgi_df = fgi_df.sort_values('Date', ascending=True)
    fgi_date_list = fgi_df['Date'].tolist()
    value_list = fgi_df['Value'].tolist()
    fgi_dict = {k: v for k, v in zip(fgi_date_list, value_list)}
    conn.close()

    operation_record_list = list()
    accum_invest_usdt = init_vest_usdt
    accum_usdt_asset = init_vest_usdt
    accum_btc_asset = 0
    rest_asset_value = 0
    fixed_invest_days = 0

    fgi_fixed_invest_usdt = 100
    dynamic_invest_usdt_everyday = max(
        init_vest_usdt / max_batch_days, fixed_invest_days)

    max_lower_days = 0
    current_lower_days = 0
    lower_days_list = list()

    min_profit_rate = 0.0
    max_profit_rate = 0.0
    profit_rate_list = list()
    profit_rate_list.append(0.0)

    for i in range(1, len(date_list)):
        operation_record = dict()
        if math.isnan(ma_price_list[i-1]):
            profit_rate_list.append(0.0)
            continue

        fgi_value = fgi_dict[date_list[i]] if date_list[i] in fgi_dict else 50

        # 价格上穿均线
        if price_list[i-1] <= ma_price_list[i-1] and price_list[i] > ma_price_list[i]:
            accum_btc_asset += accum_usdt_asset / price_list[i]
            accum_usdt_asset = 0
            operation_record['date'] = date_list[i]
            operation_record['operation'] = 'STOP_FIXED_INVEST'
            operation_record['price'] = price_list[i]
            operation_record['asset'] = accum_btc_asset * price_list[i]
            operation_record_list.append(operation_record)

            if current_lower_days > max_lower_days:
                max_lower_days = current_lower_days
            if current_lower_days > 10:
                lower_days_list.append(current_lower_days)
            current_lower_days = 0

        # 价格下穿均线
        if price_list[i-1] >= ma_price_list[i-1] and price_list[i] < ma_price_list[i]:
            if accum_btc_asset == 0:
                profit_rate_list.append(0.0)
                continue

            accum_usdt_asset += accum_btc_asset * price_list[i]
            accum_btc_asset = 0
            operation_record['date'] = date_list[i]
            operation_record['operation'] = 'SELL_AND_START_FIXED_INVEST'
            operation_record['price'] = price_list[i]
            operation_record['asset'] = accum_usdt_asset
            operation_record_list.append(operation_record)
            current_lower_days = 0
            dynamic_invest_usdt_everyday = max(
                accum_usdt_asset / max_batch_days, fixed_invest_days)

        # FGI增强定买
        if fgi_enhancement and fgi_value <= 25:
            if accum_usdt_asset >= fgi_fixed_invest_usdt:
                accum_btc_asset += (fgi_fixed_invest_usdt / price_list[i])
                accum_usdt_asset -= fgi_fixed_invest_usdt
            else:
                accum_invest_usdt += fgi_fixed_invest_usdt
                accum_btc_asset += (fgi_fixed_invest_usdt / price_list[i])

        # FGI增强定卖
        if fgi_enhancement and fgi_value >= 63:
            if accum_btc_asset > fgi_fixed_invest_usdt / price_list[i]:
                accum_btc_asset -= (fgi_fixed_invest_usdt / price_list[i])
                accum_usdt_asset += fgi_fixed_invest_usdt

        # 价格低于均线时定投
        if price_list[i] < ma_price_list[i]:

            # 价格均线差值百分比增强
            dr_enhancement_times = 1
            diff_rate = min(
                price_list[i] - ma_price_list[i], 0.0) / ma_price_list[i] * 100
            if dr_enhancement and price_list[i] - ma_price_list[i] < 0.0:
                for k, x_area in enumerate([10, 20, 30, 40, 50, 100]):
                    if abs(diff_rate) >= x_area:
                        continue
                    dr_enhancement_times = k + 1
                    break

            enhance_invest_usdt = dynamic_invest_usdt_everyday * dr_enhancement_times
            current_invest_usdt = min_fixed_invest_usdt
            if batch_fixed_invest and accum_usdt_asset > enhance_invest_usdt:
                current_invest_usdt = enhance_invest_usdt + min_fixed_invest_usdt
                accum_usdt_asset -= enhance_invest_usdt

            accum_invest_usdt += min_fixed_invest_usdt
            accum_btc_asset += current_invest_usdt / price_list[i]
            fixed_invest_days += 1
            current_lower_days += 1

        # 价格高于均线且高于上次牛市峰值指定倍数价格时定卖
        previous_peak_price = 20000
        base_peak_times = 2.5
        step_price = 4000
        sell_price_thresh = previous_peak_price * base_peak_times
        if (price_list[i] > ma_price_list[i]) and (price_list[i] > sell_price_thresh):
            sell_share_times = float(
                price_list[i] - sell_price_thresh) / step_price
            sell_btc_amount = accum_btc_asset * 0.01 * sell_share_times
            accum_btc_asset = accum_btc_asset - sell_btc_amount
            accum_usdt_asset = accum_usdt_asset + \
                sell_btc_amount * price_list[i]
            # print(f'date : {date_list[i]}, fixed_sell_usdt : {sell_btc_amount * price_list[i]}, price : {price_list[i]}, accum_btc_asset : {accum_btc_asset}')

        rest_asset_value = accum_usdt_asset + accum_btc_asset * price_list[i]
        profit_rate = (rest_asset_value - accum_invest_usdt) / \
            accum_invest_usdt * 100
        profit_rate_list.append(profit_rate)
        if profit_rate > max_profit_rate:
            max_profit_rate = profit_rate
        if profit_rate < min_profit_rate:
            min_profit_rate = profit_rate

    # print(f'max_lower_days : {max_lower_days}')
    print(f'mean_lower_days : {sum(lower_days_list) / len(lower_days_list)}')
    print(f'final_invest_days : {fixed_invest_days}')
    print(f'final_invest_usdt : {accum_invest_usdt}')
    print(f'max_profit_rate : {max_profit_rate:.2f}%')
    print(f'min_profit_rate : {min_profit_rate:.2f}%')
    print(
        f'final_rest_asset_usdt : {rest_asset_value}, profit_rate : {profit_rate:.2f}%')

    # for record in operation_record_list: print(record)
    return profit_rate_list

# 计算双均线策略收益
def calculate_double_ma_strategy_profit(dataframe, short_ma, long_ma, init_invest_usdt):
    date_list = dataframe['Date'].tolist()
    price_list = dataframe['Close'].tolist()
    short_ma_price_list = (dataframe.rolling(window=int(short_ma))[
                           ['Close']].mean())['Close'].tolist()
    long_ma_price_list = (dataframe.rolling(window=int(long_ma))[
        ['Close']].mean())['Close'].tolist()

    operation_record_list = list()
    accum_usdt_asset = init_invest_usdt
    accum_btc_asset = 0

    for i in range(1, len(date_list)):
        operation_record = dict()
        previous_diff = short_ma_price_list[i-1] - long_ma_price_list[i-1]
        current_diff = short_ma_price_list[i] - long_ma_price_list[i]

        # 短周期均线上穿长周期均线
        if previous_diff <= 0.0 and current_diff > 0.0:
            accum_btc_asset = accum_usdt_asset / price_list[i]
            operation_record['date'] = date_list[i]
            operation_record['operation'] = 'BUY'
            operation_record['price'] = price_list[i]
            operation_record['asset'] = accum_usdt_asset
            operation_record_list.append(operation_record)

        # 短周期均线下穿长周期均线
        if previous_diff >= 0.0 and current_diff < 0.0:
            if accum_btc_asset == 0:
                continue
            accum_usdt_asset = accum_btc_asset * price_list[i]
            operation_record['date'] = date_list[i]
            operation_record['operation'] = 'SELL'
            operation_record['price'] = price_list[i]
            operation_record['asset'] = accum_usdt_asset
            operation_record_list.append(operation_record)

    print(f'total_rest_asset_usdt : {accum_usdt_asset}')
    for record in operation_record_list:
        print(record)

# 计算价值平均定投策略收益
def calculate_va_fixed_invest_strategy_profit(dataframe, invest_usdt_everyday):
    date_list = dataframe['Date'].tolist()
    price_list = dataframe['Close'].tolist()

    va_value_list = list()
    for i in range(1, len(date_list) + 1):
        va_value_list.append(invest_usdt_everyday * i)

    rest_usdt_value_list = list()
    accum_invest_usdt_list = list()
    rest_btc_value = 0
    accum_invest_usdt = 0
    accum_take_usdt = 0

    for i in range(len(date_list)):
        diff_usdt_value = rest_btc_value * price_list[i] - va_value_list[i]
        if diff_usdt_value < 0:
            accum_invest_usdt += abs(diff_usdt_value)
            rest_btc_value += abs(diff_usdt_value) / price_list[i]
        if diff_usdt_value > 0:
            accum_take_usdt += abs(diff_usdt_value)
            rest_btc_value -= abs(diff_usdt_value) / price_list[i]
        rest_usdt_value_list.append(va_value_list[i] + accum_take_usdt)
        accum_invest_usdt_list.append(accum_invest_usdt)

    print(f'accum_invest_usdt : {accum_invest_usdt}')
    print(f'rest_asset_value : {rest_usdt_value_list[-1]}')
    print(
        f'final_profit_rate : {(rest_usdt_value_list[-1] - accum_invest_usdt) / accum_invest_usdt * 100}%')

    xticks = list(range(0, len(date_list), max([len(date_list) // 40, 1])))
    xlabels = [date_list[i] for i in xticks]

    figure = plt.figure(figsize=(38, 8))
    gs = GridSpec(1, 1)
    ax = plt.subplot(gs[0, 0])
    ax.plot(date_list, price_list, label='btc_price')
    ax.plot(date_list, rest_usdt_value_list, label='asset_value_cumsum')
    ax.plot(date_list, accum_invest_usdt_list, label='invest_value_cumsum')

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=70)

# 计算 Vegas 隧道策略收益
def calculate_vegas_tunnel_strategy_profit(dataframe, init_invest_usdt):
    date_list = dataframe['Date'].tolist()
    price_list = dataframe['Close'].tolist()

    xticks = list(range(0, len(date_list), max([len(date_list) // 40, 1])))
    xlabels = [date_list[i] for i in xticks]

    figure = plt.figure(figsize=(38, 8))
    gs = GridSpec(1, 1)
    ax = plt.subplot(gs[0, 0])
    ax.plot(date_list, price_list, label='Price')

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=70)

    ema_dict = dict()
    ema_window_list = [12, 144, 169, 576, 676]
    for ema_window in ema_window_list:
        ema = (dataframe.ewm(span=ema_window, adjust=False).mean())[
            'Close'].tolist()
        ema_dict[str(ema_window)] = ema
        ax.plot(date_list, ema, label='EMA'+str(ema_window))

    ax.legend(loc=1)

    operation_record_list = list()
    accum_usdt_asset = init_invest_usdt
    accum_btc_asset = 0
    previous_operation = 'SELL'

    for i in range(700, len(date_list)):
        operation_record = dict()
        previous_price = price_list[i-1]
        previous_ema12 = ema_dict['12'][i-1]
        previous_ema144 = ema_dict['144'][i-1]
        previous_ema169 = ema_dict['169'][i-1]
        previous_ema576 = ema_dict['576'][i-1]
        previous_ema676 = ema_dict['676'][i-1]

        current_price = price_list[i]
        current_ema12 = ema_dict['12'][i]
        current_ema144 = ema_dict['144'][i]
        current_ema169 = ema_dict['169'][i]
        current_ema576 = ema_dict['576'][i]
        current_ema676 = ema_dict['676'][i]

        max_previous_long_ema = max(previous_ema144, previous_ema169)
        min_previous_long_ema = min(previous_ema144, previous_ema169)
        max_current_long_ema = max(current_ema144, current_ema169)
        min_current_long_ema = min(current_ema144, current_ema169)

        max_previous_assist_ema = max(previous_ema576, previous_ema676)
        min_previous_assist_ema = min(previous_ema576, previous_ema676)
        max_current_assist_ema = max(current_ema576, current_ema676)
        min_current_assist_ema = min(current_ema576, current_ema676)

        previous_buy_signal_matched = min(
            previous_price, previous_ema12) > max_previous_long_ema
        current_buy_signal_matched = min(current_price, current_ema12) > max(
            max_current_long_ema, max_current_assist_ema)

        previous_sell_signal_matched = max(
            previous_price, previous_ema12) < min_previous_long_ema
        current_sell_signal_matched = max(current_price, current_ema12) < min(
            min_current_long_ema, min_current_assist_ema)

        # 价格上穿隧道出现买入信号
        if previous_operation == 'SELL' and current_buy_signal_matched:
            accum_btc_asset = accum_usdt_asset / price_list[i]
            operation_record['date'] = date_list[i]
            operation_record['operation'] = 'BUY'
            operation_record['price'] = price_list[i]
            operation_record['asset'] = accum_usdt_asset
            operation_record_list.append(operation_record)
            previous_operation = 'BUY'

        # 价格下穿均线
        if previous_operation == 'BUY' and current_sell_signal_matched:
            if accum_btc_asset == 0:
                continue
            accum_usdt_asset = accum_btc_asset * price_list[i]
            operation_record['date'] = date_list[i]
            operation_record['operation'] = 'SELL'
            operation_record['price'] = price_list[i]
            operation_record['asset'] = accum_usdt_asset
            operation_record_list.append(operation_record)
            previous_operation = 'SELL'

    print(f'total_rest_asset_usdt : {accum_usdt_asset}')
    for record in operation_record_list:
        print(record)
