import pandas as pd

class Logger():
    def __init__(self, save_path):
        """
        What we want in this class:
        something that we can say 'here take this number for this variable
        and save it in the csv' without necessarily holding the csv in memory/
        """
        self.save_path = save_path

    def load_csv_df(self):
        load_csv_df = pd.read_csv(self.save_path)

    def save_csv(self, csv_df):
        csv_df.to_csv(self.save_path)

    def log(self, variable_name, data):
