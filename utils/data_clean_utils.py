# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 14:20:51 2025

@author: RW
"""
import pandas as pd

# clean_call_options_data
def clean_call_options_data(file_path):
    """
    Process the Excel file to calculate and extract specific columns.

    Parameters:
    - file_path (str): Path to the Excel file.

    Returns:
    - pd.DataFrame: Filtered DataFrame with columns 'strike_price', 'trading_volume', and 'implied_volatility'.
    """
    try:
        # Import the Excel file into a DataFrame
        df = pd.read_excel(file_path)
        
        # Print df for debugging
        print(df)

        # Add 'strike_price' column: extract last 4 digits of '名称', convert to float, divide by 1000
        df['strike_price'] = df['行权']

        # Add 'trading_volume' column: copy values from '总量'
        df['trading_volume'] = df['持仓量']
        
        # Add 'call_price' column: copy values from '总量'
        df['call_price'] = pd.to_numeric(df['最新'], errors='coerce')

        # Add 'implied_volatility' column: copy values from '隐波%'
        df['implied_volatility'] = df['隐含波动%']
        
        # Convert the '到期日' column to datetime format
        df['expiration_date'] = pd.to_datetime(df['到期日'], format='%Y%m%d')

        # Strip the time part by converting the datetime to date (only year-month-day)
        df['expiration_date'] = df['expiration_date'].dt.normalize()  # Resets time to midnight

        # Calculate the difference in days between 'expiration_date' and today's date
        today = pd.to_datetime('today').normalize()  # Get today's date and reset time to midnight

        # Calculate the difference in days
        df['days_to_expiration'] = (df['expiration_date'] - today).dt.days

        # Filter the DataFrame to include only the required columns (if necessary)
        # filtered_df = df[['strike_price', 'trading_volume', 'implied_volatility']]

        return df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

'''TEST

# File path
file_path = 'D:\\Options\\kc50etf_call_20250208.xlsx'

# Clean call options data
df = clean_call_options_data(file_path)

# Display the resulting DataFrame
print(df)

'''




