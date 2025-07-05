# Define a class
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

class datavisualization:
    def __init__(self, data):
        self.salesdf = data      # Attribute

    def data_pivot_table(self, data, group_by_cols, value_col, agg_func, row_col, column, values):
        grouped = data.groupby(group_by_cols)[value_col].agg(agg_func).reset_index()
        # Group by State and Group, and sum Sales
        ##grouped = df.groupby(['State', 'Group'])['Sales'].sum().reset_index()
        # Pivot to have demographic groups as columns
        pivot_table = grouped.pivot(index=row_col, columns=column, values=values).fillna(0)
        # Optional: Add Total Sales per State
        pivot_table['Total'] = pivot_table.sum(axis=1)
        return pivot_table;

    def sales_by_chart(self, data, column, xlabel, ylabel, chart_title, legend_title):
        ax  = data.drop(columns=column).plot(kind='bar', stacked=True, title=chart_title, figsize=(10, 6))
        # Set labels
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Annotate total sales on top of bars
        totals = data[column]
        
        # Annotate total sales above bars
        for idx, total in enumerate(data[column]):
            ax.text(idx, total + 500000, f'{total/1e6:.1f}M', ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Format y-axis to show full numbers (avoid scientific notation)
        ax = plt.gca()  # Get current axes
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
        
        plt.tight_layout()
        plt.show()

    def heatmap_simple(self, pivot_table, chart_title):              
        # Convert pivot table to numpy array
        data = pivot_table.values
        
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(data, cmap='YlGn')

        # Set ticks and labels
        ax.set_xticks(np.arange(pivot_table.shape[1]), labels=pivot_table.columns, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticks(np.arange(pivot_table.shape[0]), labels=pivot_table.index)

        ##ax.set_xticklabels(pivot_table.columns)
        ##ax.set_yticklabels(pivot_table.index)
        
        # Show all ticks and label them with the respective list entries
        ##ax.set_xticks(range(len(farmers)), labels=farmers,
        ##              rotation=45, ha="right", rotation_mode="anchor")
        ##ax.set_yticks(range(len(vegetables)), labels=vegetables)
        
        # Loop over data dimensions and create text annotations.
        for i in range(pivot_table.shape[0]):
            for j in range(pivot_table.shape[1]):
                value = f"{data[i, j] // 1000}k"
                text = ax.text(j, i, value, ha="center", va="center", color="black")
        
        ax.set_title(chart_title)
        fig.tight_layout()
        plt.show()
        
    def heatmap_complex(self, pivot_table, xlabel, ylabel, column, chart_title):  
        plt.figure(figsize=(10, 3))
        sns.heatmap(pivot_table.drop(columns=column), annot=True, fmt=".0f", cmap="YlGnBu")
        plt.title(chart_title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.show()
    
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


