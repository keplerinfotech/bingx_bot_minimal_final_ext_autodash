class DataSource:
    def __init__(self, df):
        self.df = df

    def iter_bars(self):
        for row in self.df.itertuples():
            yield row
