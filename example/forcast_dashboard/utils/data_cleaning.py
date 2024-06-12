import pandas as pd

class DataCleaner:
    def __init__(self, df):
        self.df = df
    
    def clean_data(self, date_column):
        cleaned_df = self.df.copy()
        cleaned_df[date_column] = pd.to_datetime(cleaned_df[date_column], errors='coerce')
        cleaned_df.dropna(subset=[date_column], inplace=True)
        cleaned_df[date_column] = cleaned_df[date_column].dt.strftime('%Y-%m-%d')
        return cleaned_df

    def filter_data(self, column, value):
        return self.df[self.df[column] == value]

    def aggregate_data(self, date_column, target_column, additional_columns):
        self.df[date_column] = pd.to_datetime(self.df[date_column])
        grouped_df = self.df.groupby(date_column).agg(
            {target_column: 'sum', **{col: 'sum' for col in additional_columns}}).reset_index()
        grouped_df = grouped_df.rename(columns={target_column: 'y'})
        return grouped_df
        # self.df[date_column] = pd.to_datetime(self.df[date_column])
        # grouped_df = self.df.groupby(date_column).agg(
        #     {target_column: 'sum', **{col: 'sum' for col in additional_columns}}).reset_index()
        # return grouped_df
