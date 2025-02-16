# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 14:09:57 2025

@author: RW
"""
import pandas as pd
from utils.data_clean_utils import clean_call_options_data
from utils.implied_volatility import calculate_general_implied_volatility
from utils.ploting import plot_sell_call_profit_loss
from utils.naked_short_call_evaluation import naked_short_call_evaluation
from utils.covered_call_write_evaluation import covered_call_write_evaluation

import os

# File path
file_path = 'D:\\Options\\h300etf250213.xlsx'

def generate_strategy_output_file_path(file_path, strategy_name):
    """
    Generate the output file path based on the input file path and strategy name.

    Parameters:
    - file_path: str, the path to the input file.
    - strategy_name: str, the name of the strategy to include in the output file name.

    Returns:
    - str, the path to the output file.
    """
    # Split the file path into directory, file name, and extension
    directory, full_file_name = os.path.split(file_path)
    file_name, file_extension = os.path.splitext(full_file_name)

    # Generate the new file name with strategy name and "evaluation" added
    output_file_name = f"{file_name}_{strategy_name}_{file_extension}"

    # Construct the output file path
    output_file_path = os.path.join(directory, output_file_name)

    return output_file_path


# Clean call options data
df = clean_call_options_data(file_path)

if df is not None:
    print(df.head())
    
    stock_price = 3.889
    max_distance = 0.25
    general_iv = calculate_general_implied_volatility(df, stock_price, max_distance)
    
    if general_iv:
        print(f"General Implied Volatility: {general_iv:.4f}")
        sigma = general_iv / 100  # Convert to decimal form
        df_calls_evaluation = covered_call_write_evaluation(stock_price, df, sigma)
        print(df_calls_evaluation)
        
# File path to save the evaluation
output_file_path = generate_strategy_output_file_path(file_path,
                                                      strategy_name="covered_call_write_evaluation")

# Export to Excel
if df_calls_evaluation is not None:
    try:
        df_calls_evaluation.to_excel(output_file_path, index=False)
        print(f"Results successfully saved to {output_file_path}")
    except Exception as e:
        print(f"Error saving results to Excel: {e}")
else:
    print("No evaluation data to save.")