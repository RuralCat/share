
import tushare as ts
import datetime as dt


class Stock(object):
    """
    single stock
    """
    def __init__(self, code):
        assert isinstance(code, str) and len(code) == 6
        self.code = code
        self.updated_date = ''