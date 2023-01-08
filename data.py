import numpy
import pandas
import tabulate as tb


class Data:
    def __init__(self, fn):
        self.dataframe = pandas.read_parquet(fn)

    def print(self, tabulate, size):
        if tabulate:
            print(tb.tabulate(self.dataframe[0:size], headers='keys', tablefmt='psql'))