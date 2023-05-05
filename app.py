import schedule
import requests
import json
import time

import utilities as utils
import strategies as strat

from datetime import datetime, timedelta

class BitPusher:
    def __init__(self, msg):
        self.base_url = 'https://qyapi.weixin.qq.com/cgi-bin/gettoken?'
        self.req_url = 'https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token='
        self.corpid = utils.config.get('WeCom', 'corp_id')
        self.corpsecret = utils.config.get('WeCom', 'corp_secret')
        self.agentid = utils.config.get('WeCom', 'agent_id')
        self.usr = utils.config.get('WeCom', 'user_id')
        self.msg = msg

    def get_access_token(self):
        urls = self.base_url + 'corpid=' + self.corpid + '&corpsecret=' + self.corpsecret
        resp = requests.get(urls).json()
        access_token = resp['access_token']
        return access_token

    def send_message(self):
        data = self.get_message()
        req_urls = self.req_url + self.get_access_token()
        res = requests.post(url=req_urls, data=data)
        print(res.text)

    def get_message(self):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data = {
            "touser": self.usr,
            "toparty": "@all",
            "totag": "@all",
            "msgtype": "textcard",
            "agentid": self.agentid,
            "textcard": {
                "title" : "Timed Broadcast",
                "description" : f"<div class=\"gray\">{current_time}</div> <div class=\"highlight\">{self.msg}</div>",
                "url" : "URL"
            },
            "safe": 0,
            "enable_id_trans": 0,
            "enable_duplicate_check": 0,
            "duplicate_check_interval": 1800
        }
        data = json.dumps(data)
        return data


def send_message():
    utils.update_data_to_database()

    current_btc_price = float(utils.get_current_price('BTCUSDT')['price'])
    current_fgi_index = utils.get_fgi_index(utils.get_latest_date())
    content = f'FGI Index : {current_fgi_index["value"]}, {current_fgi_index["class"]}\n'
    content = content + f'BTCUSDT Price : {round(current_btc_price, 2)}'

    bit_pusher = BitPusher(msg=content)
    bit_pusher.send_message()

if __name__ == '__main__':
    later_time = (datetime.now() + timedelta(seconds=5)).strftime("%H:%M:%S")
    schedule.every().day.at('08:00').do(send_message)
    while True:
        schedule.run_pending()
        time.sleep(1)

    # df = utils.read_data_to_dataframe("btcusdt", "2018-11-17", "2023-05-03")
    # profit_rate_list = strat.calculate_ma_fixed_invest_strategy_profit(df, 120, 10000, 40, 250, True, False, True)
    # utils.show_data_to_plot(df, "Close", [120], profit_rate_list)

    # bit_pusher = BitPusher(usr='HuangGuanXiong', msg="你好\n这是一条测试信息1\n这是一条测试信息2")
    # bit_pusher.send_message()
