# Define a class
import pandas as pd
import numpy as np

class dataanalysis:
    def __init__(self, data):
        self.salesdf = data      # Attribute

    def data_analysis_group(self, data, group_by_cols, value_col, agg_func):
        grouped = data.groupby(group_by_cols)[value_col].agg(agg_func).reset_index()
        return grouped;

    def descriptive_statistic(self, data, group_by_cols, value_cols):
        ## checking for standard deviation applicable for the groups and filter only applicable rows
        filtered_df = data.groupby(group_by_cols).filter(lambda x: x[value_cols].notna().sum() > 1)

        if filtered_df.empty:
            print("No groups with more than one non-null row found.")
            filtered_df = data;

        filtered_finaldf = filtered_df.groupby(group_by_cols).filter(lambda x: not x[value_cols].dropna().mode().empty)

        if filtered_finaldf.empty:
            print("No groups with non-null row found before computing mode.")
            result_stats = filtered_df
        else:
            # Proceed with aggregation or other operations
            result_stats = filtered_finaldf.groupby(group_by_cols)[value_cols].agg(mean='mean', median='median', std='std', mode=lambda x: x.mode().iloc[0]).reset_index()
            
        return result_stats;

    def mini_max_by_group(self, data, group_by_cols, value_cols):
        # Group by Group and sum the sales
        grouped_sum = data.groupby(group_by_cols)[value_cols].sum()
        # Identify groups with highest and lowest sales
        highest_by_group = grouped_sum.idxmax()
        lowest_by_group = grouped_sum.idxmin()
        return (lowest_by_group, highest_by_group, grouped_sum);

    def datetime_feature_extraction(self, data, time_col):
        data['Day'] = pd.to_datetime(data[time_col]).dt.day
        data['Week'] = pd.to_datetime(data[time_col]).dt.to_period('W')
        data['Month'] = pd.to_datetime(data[time_col]).dt.month
        data['Quarter'] = pd.to_datetime(data[time_col]).dt.to_period('Q')
        data['Year'] = pd.to_datetime(data[time_col]).dt.year
        ##df['Month'] = pd.to_datetime(df['Date']).dt.to_period('M')
        ###df['Year'] = pd.to_datetime(df['Date']).dt.to_period('Y')
        return data;

    def time_based_analysis(self, data, group_by_cols, value_cols):
        grouped = data.groupby(group_by_cols)[value_cols].agg(['sum','mean','std','count']).reset_index();
        grouped.columns = [group_by_cols,'Total Units','Average Units','Units Std Dev','Units Count','Total Sales','Average Sales','Sales Std Dev','Sales Count']
        return grouped


