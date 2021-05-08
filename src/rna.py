import pandas as pd


class RNA:
    def __init__(self, data=None, folders=1, rate=1, hidden=None):
        self.data = data
        self.folders = folders
        self.rate = rate
        self.hidden = hidden

    def load_csv(self, path):
        self.data = pd.read_csv(path)

    def load_excel(self, path):
        self.data = pd.read_excel(path)

