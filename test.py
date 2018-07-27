import tushare as ts
import numpy as np
import time
import os
import pickle
import matplotlib.pyplot as plt

def ema(data, n=24, val_name="close"):
    import numpy as np
    '''
        指数平均数指标 Exponential Moving Average
        Parameters
        ------
          data:pandas.DataFrame
                      通过 get_h_data 取得的股票数据
          n:int
                      移动平均线时长，时间单位根据data决定
          val_name:string
                      计算哪一列的列名，默认为 close 收盘值
        return
        -------
          EMA:numpy.ndarray<numpy.float64>
              指数平均数指标
    '''

    prices = data[val_name]

    EMA = []

    for index in range(len(prices)):
        c_p = prices[-(index+1)]
        if index == 0:
            past_ema = c_p
        else:
            # Y=[2*X+(N-1)*Y’]/(N+1)
            today_ema = (2 * c_p + (n - 1) * past_ema) / (n + 1)
            past_ema = today_ema

        EMA.append(past_ema)

    return np.asarray(EMA)

def show_current_market():
    # a = ts.get_today_all()
    stf = lambda v: float(v)
    codes = ['601990', '601330', '603348', '300058']
    while True:
        for code in codes:
            s1 = ts.get_realtime_quotes(code)
            current_p = stf(s1['price'])
            open_p = stf(s1['open'])
            preclose_p = stf(s1['pre_close'])
            change = (current_p - preclose_p) / preclose_p * 100
            print('code: {}, cp: {}, change: {:.2F}'.format(code, current_p, change))
        print('')

        inds = [0, 12, 17]
        df = ts.get_index()
        for ind in inds:
            print('code: {}, change: {:.2F}'.format(df['code'][ind], df['change'][ind]))
        for _ in range(5): print('')

        # pyautogui.hotkey('ctrl', 'alt', 'l')
        time.sleep(5)

def plot_ema(code):
    d_path = 'data/{}.pic'.format(code)
    if os.path.exists(d_path):
        with open(d_path, 'rb') as f:
            d = pickle.load(f)
    else:
        d = ts.get_h_data(code)
        with open(d_path, 'wb') as f:
            pickle.dump(d, f)
    d_ema = ema(d)
    print(d_ema)
    plt.plot(d_ema)
    plt.show()

url = 'http:/vip.stock.finance.sina.com.cn/corp/go.php/vMS_FuQuanMarketHistory/stockid/601990.phtml?year=2018&jidu=3'
if __name__ == '__main__':
    show_current_market()



