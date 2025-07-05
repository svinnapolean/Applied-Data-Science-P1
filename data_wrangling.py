# Define a class
import numpy as np

class datawrangling:
    def __init__(self, data):
        self.salesdf = data      # Attribute
        
    def datainspection(self):       # Method
        # Inspecting the first few rows of the DataFrame
        print(f"\nInspecting the first few rows of the DataFrame.")
        print(self.salesdf.head())

        # Displaying the last few rows of the DataFrame
        print(f"\nDisplaying the last few rows of the DataFrame")
        print(self.salesdf.tail())

        # Providing information about the DataFrame, including data types and non-null counts
        print(f"\nProviding information about the DataFrame, including data types and non-null counts")
        print(self.salesdf.info())

        # Displaying descriptive statistics of the DataFrame, such as mean, std, min, max, and so on.
        print(f"\nDisplaying descriptive statistics of the DataFrame, such as mean, std, min, max, and so on.")
        print(self.salesdf.describe())

        # Displaying datatypes of the columns
        print(f"\nDisplaying datatypes of the columns")
        self.salesdf.dtypes

    def clean_null_records(self):
        # Checking for missing values
        ## missing_values = self.df.isnull().sum()
        isna_values = self.salesdf.isna().sum()
        isnotna_values = self.salesdf.notna().sum()
        duplicate_count = self.salesdf.duplicated().sum()
        
        if self.salesdf.isna().any().any():
            df_cleaned = self.salesdf.dropna()
            print(f"\nMissing values found and cleaned (number of null rows) : {df_cleaned}")
            self.salesdf = df_cleaned
        else:
            print(f"\nMissing values not found and cleaned (number of null rows).")

        if self.salesdf.notna().any().any():
            print(f"\nNon-null values per column :\n")
            print(isnotna_values)
            
        else:
            print(f"\nAll the records are null.")
    
        if duplicate_count > 0:
            df_no_duplicates = self.salesdf.drop_duplicates()
            print(f"\nDuplicated values found and cleaned (number of duplicated rows) : {df_no_duplicates}")
            self.salesdf = df_no_duplicates
        else:
            print(f"\nDuplicated values not found.")
        
        return self.salesdf;

    def data_normalization(self, data):
        self.salesdf = data

        df_normalized = self.salesdf.copy()
 
        if 'Sales' in df_normalized.columns:
            # Use the natural logarithm to create a new feature 'Log_Sales'
            df_normalized['Log_Sales'] = df_normalized['Sales'].apply(lambda x: np.log(x))
        
            # Normalize 'Sales' column and create a new feature 'Normalized_Sales'
            df_normalized['Normalized_Sales'] = (df_normalized['Sales'] - df_normalized['Sales'].min()) / (df_normalized['Sales'].max() - df_normalized['Sales'].min())
        
            # Displaying the DataFrame with the new features
            print("DataFrame with new features:")
            print(df_normalized)
            self.salesdf = df_normalized
        else:
            print("The 'Sales' column does not exist in the DataFrame.")
            
        return self.salesdf;

    def data_insights_group(self, data):
        avg_sales = data.groupby(['State','Time','Group'])['Sales'].mean().reset_index()
        sum_sales = data.groupby(['State','Time','Group'])['Sales'].sum().reset_index()
        count_sales = data.groupby(['State','Time','Group'])['Sales'].count().reset_index()
        print("\nAverage Sales by State, Time, and Group:\n", avg_sales)
        print("\nTotal Sales by State, Time, and Group:\n", sum_sales)
        print("\nNumber of Transactions by State, Time, and Group:\n", count_sales)
        ##print("\nAverage Sales:\n", avg_sales.to_string(index=False))
        ##print("\nTotal Sales:\n", sum_sales.to_string(index=False))
        ##print("\nTransaction Count:\n", count_sales.to_string(index=False))
    
    def data_insights_group(self, data, group_by_cols, value_col, agg_func):
        grouped = data.groupby(group_by_cols)[value_col].agg(agg_func).reset_index()
        return grouped
