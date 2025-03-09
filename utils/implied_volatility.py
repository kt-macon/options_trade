# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 14:46:42 2025

@author: RW
"""
import pandas as pd

# calculate_general_implied_volatility

def calculate_general_implied_volatility(df, stock_price, max_distance):
    """
    Calculate the general implied volatility of a stock using a DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'strike_price', 'trading_volume', and 'implied_volatility'.
    - stock_price (float): Current price of the stock.
    - max_distance (float): The threshold for distance factor calculation.

    Returns:
    - float: General implied volatility of the stock.
    """
    if df['trading_volume'].sum() == 0:
        raise ValueError("Total trading volume cannot be zero.")

    # Calculate volume_factor
    df['volume_factor'] = df['trading_volume'] / df['trading_volume'].sum()

    # Calculate distance and distance_factor
    df['distance'] = (df['strike_price'] - stock_price) / stock_price
    df['distance_factor'] = df['distance'].apply(
        lambda d: ((d - max_distance)**2 / max_distance**2) if abs(d) <= max_distance else 0
    )

    # Weighted implied volatility
    df['weighted_iv'] = df['volume_factor'] * df['distance_factor'] * df['implied_volatility']
    df['weight'] = df['volume_factor'] * df['distance_factor']

    # Calculate general implied volatility
    numerator = df['weighted_iv'].sum()
    denominator = df['weight'].sum()
   
    if denominator == 0:
        raise ValueError("Sum of weighted distance factors is zero, cannot divide by zero.")
           
    return numerator / denominator