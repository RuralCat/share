
import numpy as np
import logging
from stock import Stock
from stock import EMA
from enum import Enum

# enum
class OperationType(Enum):
    INVALID = 0
    VALID = 1

class Stategy():
    def __init__(self):
        self.reset()

    def reset(self):
        self.buy_price = []
        self.sale_price = []
        self.ope_code = []
        self.ope_day = []
        self.nonope_code = []

    def run_strategy(self, strategy_func, code_list, *args):
        for code in code_list:
            stock = code_list[code]
            isinstance(stock, Stock)
            if stock.days_num > 0 and stock.trading_day[0] > '2018-01-01':
                bp, sp, ope, buy_day = strategy_func(code_list[code], *args)
                if ope:
                    self.ope_code.append(code)
                    self.ope_day.append(buy_day)
                    self.buy_price.append(bp)
                    self.sale_price.append(sp)
                else:
                    self.nonope_code.append(code)

    def logging(self, show_ope=True, show_nonope=False):
        # compute changes
        bps = np.asarray(self.buy_price)
        sps = np.asarray(self.sale_price)
        changes = 100 * (sps - bps) / bps
        mean_earning = np.mean(changes)
        win_rate = np.sum(sps > bps) / len(bps) * 100
        print('operated stocks : {}'.format(len(bps)))
        print('mean earning : {:.2F}'.format(mean_earning))
        print('win rate : {:.2F}'.format(win_rate))
        if show_ope:
            for code, bp, buy_day, sp in zip(self.ope_code,
                                             self.buy_price,
                                             self.ope_day,
                                             self.sale_price):
                print('{}: buy {:2F} in {}, sale {:2F}'.format(code, bp, buy_day, sp))
        if show_nonope:
            for code in self.nonope_code:
                print('{} has not been operated!'.format(code))
        # logging
        # logger = logging.getLogger()


class Operation(object):
    def __init__(self,
                 buy_price=0,
                 sell_price=0,
                 buy_day=None,
                 sell_day=None,
                 operation_type=OperationType.VALID):
        self.buy_price = buy_price
        self.sell_price = sell_price
        self.buy_day = buy_day
        self.sell_day = sell_day
        self.operation_type = operation_type

def ema_first_buy(stock, n=5, sniper_rate=1.00):
    """
    buying rules：when price undercross ema n-th line and the price firstly overcross ema5, buy on the line
    selling rules：（1）
    :param stock:
    :return: bid price, selling price, bool operation, deal date
    """
    assert isinstance(stock, Stock)
    close = np.asarray(stock.close)
    high = np.asarray(stock.high)
    low = np.asarray(stock.low)
    ema5 = np.asarray(stock.ema_price(n))
    bp = None
    sp = None
    traded_day = None
    # find first across ema5
    lema5 = ema5.copy()
    lema5[1:] = (2 * low[1:] + (n - 1) * ema5[:-1]) / (n + 1)
    hema5 = ema5.copy()
    hema5[1:] = (2 * high[1:] + (n - 1) * ema5[:-1]) / (n + 1)
    # hema5[1:] = ema5[:-1]
    ind = np.nonzero((low < lema5 * sniper_rate) & (high > ema5))[0]
    if len(ind) > 0 and ind[0] + 1 < stock.days_num and ind[0] != 0:
        bp = ((n - 1) * ema5[ind[0] - 1]) / ((n+1) / sniper_rate - 2)
        bp = close[ind[0]]
        # bp = ema5[ind[0] - 1]
        sp = high[ind[0] + 1]
        traded_day = stock.trading_day[ind[0]]
    # find first scounting point
    # ind = np.nonzero()
    ope = True if bp is not None else False

    return bp, sp, ope, traded_day

def sell_price_search(stock, buy_price, start_date, method):
    if method == 'simple max':
        sell_price = simple_max_stop_profit(stock, buy_price, start_date)

    return sell_price

def simple_max_stop_profit(stock, buy_price, start_date, max_trade_day=2):
    assert isinstance(stock, Stock)
    for i in range(max_trade_day):
        pass


if __name__ == '__main__':
    st = Stategy()
    from test import load_data
    stocks = load_data('stocks')
    rs = np.arange(17) * 0.01 + 0.92
    # rs = [0.96]
    for r in rs:
        print('sniper rate: {:.2F}'.format(r))
        st.run_strategy(ema_first_buy, stocks, 5, r)
        st.logging(show_ope=False, show_nonope=False)
        print('')
        st.reset()
