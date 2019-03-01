
import tushare as ts
import datetime as dt
import pickle
import numpy as np
import matplotlib.finance as fin
import matplotlib.pyplot as plt
from matplotlib import lines
import os
import time
from urllib.request import urlopen, Request
import re
from tushare.stock import cons as ct
import json
import pandas as pd

class Stock(object):
    """
    single stock
    """
    def __init__(self, code):
        assert isinstance(code, str) and len(code) == 6
        self.code = code
        self.updated_date = dt.datetime(2000, 1, 1)
        self.trading_day = []
        self.open = []
        self.close = []
        self.high = []
        self.low = []
        self.volume = []

    @property
    def days_num(self):
        return len(self.trading_day)

    # data
    def update(self):
        today = dt.datetime.today()
        if self.updated_date < today:
            # get new k data
            start = self.updated_date + dt.timedelta(days=1)
            data = ts.get_k_data(self.code,
                                 start.isoformat(),
                                 today.isoformat())
            self.updated_date = today
            # fill data
            if 'date' in data:
                self.trading_day.extend(data['date'])
                self.open.extend(data['open'])
                self.close.extend(data['close'])
                self.high.extend(data['high'])
                self.low.extend(data['low'])
                self.volume.extend(data['volume'])
            # indicator
            self.ema = EMA()

    def get_timeshare_data(self, date):
        # create url
        code = self.code
        if len(date) == 10: date = date[:4] + date[5:7] + date[8:10]
        symbol = 'sh%s'%code if code[:1] in ['5', '6', '9'] else 'sz%s'%code
        urls = 'http://ifzq.gtimg.cn/appstock/app/kline/mkline?param=%s,m1,%s,241&var=m1_today&r=0.10'%(symbol, date)
        data = pd.DataFrame()
        data = data.append(_get_k_data(urls, symbol=symbol, code=code),
                           ignore_index=True)
        price = data['open']
        vol = data['volume']

        return price, vol


    # indicator
    def ema_price(self, n=24):
        return  self.ema.compute_indicator(self.close, n=n)

    def realtime_ema_price(self):
        pass

    def plot_k_data(self):
        _, axes = plt.subplots(1, 1)
        fin.candlestick2_ochl(axes, self.open, self.close,
                              self.high, self.low, width=0.5,
                              colorup='r', colordown='g')
        self.ema.compute_indicator(self.close, n=24)
        self.ema.plot_indicator(axes)

        plt.show()


class Indicator(object):
    def plot_indicator(self, axes, plot_type='line'):
        if plot_type == 'line':
            axes.add_line(lines.Line2D(np.arange(len(self.value)),
                                       self.value))
        elif plot_type == 'bar':
            pass

    def compute_indicator(self, raw_data, **kwargs):
        pass

class EMA(Indicator):
    def compute_indicator(self, raw_data, **kwargs):
        assert 'n' in kwargs, 'ema need parameter named n '
        n = kwargs.get('n')
        data = []
        past_ema = raw_data[0]
        for d in raw_data:
            new_ema = (2 * d + (n - 1) * past_ema) / (n + 1)
            past_ema = new_ema
            data.append(past_ema)

        return data



def update_basic_info():
    basic_info = ts.get_stock_basics()
    with open('data/basic_info.sh', 'wb') as f:
        pickle.dump(basic_info, f)

def load_code_list(range='new', path='data/basic_info.sh'):
    if not os.path.exists(path):
        update_basic_info()
    with open('data/basic_info.sh', 'rb') as f:
        basic_info = pickle.load(f)
    if range == 'new':
        basic_info = basic_info[basic_info.timeToMarket > 20170101]
    return basic_info.index

def update_all_code(path='data/stocks.sh'):
    # read code list
    code_list = load_code_list()
    code_num = len(code_list)
    # read all k data if exist
    if os.path.exists(path):
        with open(path, 'rb') as f:
            stocks = pickle.load(f)
    else:
        stocks = {}
        for code in code_list:
            stocks[code] = Stock(code)

    # update
    for i in range(code_num):
        stock = stocks[code_list[i]]
        isinstance(stock, Stock)
        stock.update()
        print('[{}/{}] {} completed ...'.format(i + 1, code_num, code_list[i]))
    # save data
    with open(path, 'wb') as f:
        pickle.dump(stocks, f)


def _get_k_data(url, dataflag='m1',
                symbol='',
                code = '',
                index = False,
                ktype = '1',
                retry_count=3,
                pause=0.001):

    for _ in range(retry_count):
            time.sleep(pause)
            try:
                request = Request(url)
                lines = urlopen(request, timeout = 10).read()
                if len(lines) < 100: #no data
                    return None
            except Exception as e:
                print(e)
            else:
                lines = lines.decode('utf-8') if ct.PY3 else lines
                lines = lines.split('=')[0]
                reg = re.compile(r',{"nd.*?}')
                lines = re.subn(reg, '', lines)
                js = json.loads(lines[0])
                dataflag = dataflag if dataflag in list(js['data'][symbol].keys()) else ct.TT_K_TYPE[ktype.upper()]
                if len(js['data'][symbol][dataflag]) == 0:
                    return None
                if len(js['data'][symbol][dataflag][0]) == 6:
                    df = pd.DataFrame(js['data'][symbol][dataflag],
                                  columns = ct.KLINE_TT_COLS_MINS)
                else:
                    df = pd.DataFrame(js['data'][symbol][dataflag],
                                  columns = ct.KLINE_TT_COLS)
                df['code'] = symbol if index else code
                if ktype in ct.K_MIN_LABELS:
                    df['date'] = df['date'].map(lambda x: '%s-%s-%s %s:%s'%(x[0:4], x[4:6],
                                                                            x[6:8], x[8:10],
                                                                            x[10:12]))
                for col in df.columns[1:6]:
                    df[col] = df[col].astype(float)
                return df


if __name__ == '__main__':
    d = ts.get_stock_basics()
    new_info = d[d.timeToMarket > 20170101]

    # print(type(new_info.index[0]))
    # update_all_code()
