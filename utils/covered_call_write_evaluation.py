# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 13:13:47 2025

@author: RW
"""
import numpy as np
from scipy.stats import lognorm
from scipy.integrate import simps

# Modularized fee calculations
def calculate_stock_fee(price):
    """Calculate the trading fee for stocks."""
    fees = price * 10000 * 0.0001854
    return np.maximum(fees, 5)  # Vectorized operation

def calculate_option_fee():
    """Calculate the trading fee for options."""
    return 3.6


# Covered Write Strategy profit/loss
def profit_covered_write(x, stock_price, call_price, strike_price):
    """
    Calculate the profit/loss for when closing a covered call strategy.

    Parameters:
    - x: array-like, stock prices at expiration.
    - stock_price: float, purchase price of the stock.
    - call_price: float, premium received for selling the call.
    - strike_price: float, strike price of the call option.

    Returns:
    - array-like, profit/loss values.
    """
    buy_stock_fee = calculate_stock_fee(stock_price)
    sell_stock_fee = calculate_stock_fee(strike_price)
    sell_call_fee = calculate_option_fee()

    # Option exercised; stocks sold at strike price
    profit_exercised = (strike_price - stock_price + call_price) * 10000 - buy_stock_fee - sell_stock_fee - sell_call_fee
    
    # Option unexcercised: stock kept with unrealized profit/loss, option expried without value
    # Unrealized profit/loss from stock
    # Option expired, premium kept
    profit_not_exercised = (x - stock_price + call_price) * 10000 - buy_stock_fee - sell_call_fee
    
    # Use NumPy to select profits based on conditions
    profit = np.where(x > strike_price, profit_exercised, profit_not_exercised)
    return profit


# Covered call write evaluation
def covered_call_write_evaluation(stock_price, df, sigma):
    """
    Evaluate key metrics for selling call options.

    Parameters:
    - stock_price: float, current stock price.
    - df: DataFrame containing 'call_price' and 'strike_price' columns.
    - sigma: float, volatility (standard deviation) of the stock.

    Returns:
    - pd.DataFrame: Updated DataFrame with key metrics for evaluation.
    """
    #mu = np.log(stock_price) - 0.5 * sigma**2 * T
    df_calls = df.copy()
    
    # Here we scale sigma based on time to expiration for each option
    df_calls['T'] = df_calls['days_to_expiration'] / 365
    df_calls['sigma_T'] = df_calls['T'].apply(lambda T: sigma * np.sqrt(T))
    df_calls['mu'] = np.log(stock_price) - 0.5 * sigma**2 * df_calls['T']

    fee_buy_stock = calculate_stock_fee(stock_price)
    fee_sell_call = calculate_option_fee()
    fee_total = fee_buy_stock + fee_sell_call
    
    # Probability of win premium
    df_calls['premium'] = df_calls['call_price'] * 10000 - fee_sell_call
    df_calls['expire_worthless_probability (%)'] = lognorm.cdf(
        df_calls['strike_price'], df_calls['sigma_T'], 
        scale=np.exp(df_calls['mu'])) * 100

    # Break-even point calculations
    df_calls['break_even_x'] = stock_price - df_calls['call_price'] + fee_total / 10000
    # Use the scaled sigma_T for break-even probability
    df_calls['break_even_probability (%)'] = (1 - lognorm.cdf(
        df_calls['break_even_x'], df_calls['sigma_T'], 
        scale=np.exp(df_calls['mu']))) * 100

    # Max profit point calculations
    df_calls['max_profit_x'] = df_calls['strike_price']
    #df_calls['max_profit'] = (df_calls['strike_price'] - stock_price) * 10000 - calculate_stock_fee(df_calls['strike_price'])
    df_calls['max_profit'] = profit_covered_write(df_calls['strike_price'], stock_price, df_calls['call_price'], df_calls['strike_price'])
    # Use the scaled sigma_T for max profit probability
    df_calls['max_profit_probability (%)'] = (1 - lognorm.cdf(
        df_calls['max_profit_x'], df_calls['sigma_T'], 
        scale=np.exp(df_calls['mu']))) * 100
        
    

    # Calculate expected profit and profit rate
    #df_calls = calculate_expected_profit_rate(df_calls, stock_price, mu, sigma) 
    df_calls['extra_anual_return_rate_from_premium (%)'] = 0.0 # the extra return from premium when option expires worthless
    df_calls['max_profit_rate (%)'] = 0.0
    df_calls['max_profit_rate_annualized (%)'] = 0.0
    df_calls['expected_profit'] = 0.0
    df_calls['expected_profit_rate (%)'] = 0.0
    df_calls['expected_profit_rate_annualized (%)'] = 0.0
    
    num_points=1000

    for index, row in df_calls.iterrows():
        call_price = row['call_price']
        strike_price = row['strike_price']
        sigma_T = row['sigma_T']  # Scaled volatility
        mu = row['mu']
        T = row['T']
        premium = row['premium']
        
        stock_price_range = np.linspace(0, 5 * np.exp(mu), num_points)
        
        # Define pdf with the correctly scaled sigma_T
        pdf = lognorm.pdf(stock_price_range, sigma_T, scale=np.exp(mu))        

        # Calculate profit for covered call strategy
        profits = profit_covered_write(stock_price_range, stock_price, call_price, strike_price)

        # Numerically integrate to calculate expected profit
        expected_profit = simps(profits * pdf, stock_price_range)

        # Calculate cost 
        buy_stock_fee = calculate_stock_fee(stock_price)
        sell_call_fee = calculate_option_fee()
        
        cost = (stock_price - call_price) * 10000 + buy_stock_fee + sell_call_fee
        
        #cost = np.array([calculate_cost(x, stock_price, call_price, strike_price)
         #                for x in stock_price_range]).mean()
         
        # Calculate the extra return rate from the premium when the option expires worthless
        # The goal is to keep the stock and collect the premium
        # The extra return rate from premium is simply: premium / cost / time to expiration
            #extra_return_rate_from_premium = return_rate_winning_premium - return_rate_of_stock_itself
            #return_rate_winning_premium_annualized =  [(stock_price - stock_buy_price + premium)/ stock_buy_price]/T
            #return_rate_of_stock_itself_annualized = [(stock_price - stock_buy_price) / stock_buy_price] / T
            #extra_return_rate_from_premium = premium / stock_buy_price / T
        # The value * probability = expected extra_return_rate_from_premium
        extra_return_rate_from_premium = (premium / (stock_price * 10000) / T) * 100
        df_calls.at[index, 'extra_anual_return_rate_from_premium (%)'] = extra_return_rate_from_premium
        
        #Caculate max_profit_rate
        max_profit = row['max_profit'] 
        max_profit_rate = max_profit / cost *100 if cost > 0 else 0
        max_profit_rate_annualized = max_profit_rate / T 
        df_calls.at[index, 'max_profit_rate (%)'] = max_profit_rate
        df_calls.at[index, 'max_profit_rate_annualized (%)'] = max_profit_rate_annualized
       

        # Calculate expected profit rate
        expected_profit_rate = (expected_profit / cost) * 100 if cost > 0 else 0
        expected_profit_rate_annualized = expected_profit_rate / T

        df_calls.at[index, 'expected_profit'] = expected_profit
        df_calls.at[index, 'expected_profit_rate (%)'] = expected_profit_rate
        df_calls.at[index, 'expected_profit_rate_annualized (%)'] = expected_profit_rate_annualized
        

    return df_calls[['序','代码', '名称','总量','持仓量',
                    'expiration_date','days_to_expiration','隐含波动%',
                    'Delta','Gamma','Vega','Theta','Rho',
                    'strike_price','call_price',
                    'premium', 'expire_worthless_probability (%)', 'extra_anual_return_rate_from_premium (%)',
                    'break_even_x', 'break_even_probability (%)',
                    'max_profit','max_profit_probability (%)', 'max_profit_rate (%)','max_profit_rate_annualized (%)',
                    'expected_profit','expected_profit_rate (%)','expected_profit_rate_annualized (%)']]


