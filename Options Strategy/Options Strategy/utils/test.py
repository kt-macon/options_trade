# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 15:39:24 2025

@author: RW
"""
import numpy as np
import pandas as pd
from scipy.stats import lognorm
from scipy.integrate import simps
from typing import Union, Optional

# --------------------------
# 常量定义 (集中管理硬编码值)
# --------------------------
LOT_SIZE = 10000          # 每手股数
FEE_RATE_STOCK = 0.0001854 # 股票交易费率
MIN_STOCK_FEE = 5.0        # 最低股票手续费
OPTION_FEE = 3.6           # 期权固定手续费
NUM_INTEGRATION_POINTS = 500  # 积分点数优化

# --------------------------
# 手续费计算模块
# --------------------------
def calculate_stock_fee(price: Union[float, int]) -> float:
    """计算股票交易手续费，确保最低费用
    
    Args:
        price: 股票单价（必须 >= 0）
    
    Returns:
        手续费金额，不低于 MIN_STOCK_FEE
    
    Raises:
        ValueError: 如果 price 为负数或非数值类型
    """
    if not isinstance(price, (int, float)) or price < 0:
        raise ValueError("股票价格必须为非负数")
    return max(price * LOT_SIZE * FEE_RATE_STOCK, MIN_STOCK_FEE)

def calculate_option_fee() -> float:
    """期权固定手续费"""
    return OPTION_FEE

# --------------------------
# 策略盈亏计算模块
# --------------------------
def calculate_covered_call_profit(
    expiration_prices: np.ndarray,
    stock_purchase_price: float,
    call_premium: float,
    strike_price: float
) -> np.ndarray:
    """计算备兑看涨策略在到期日的盈亏
    
    Args:
        expiration_prices: 到期日股票价格数组
        stock_purchase_price: 股票买入价格
        call_premium: 期权权利金（每股）
        strike_price: 行权价
    
    Returns:
        各价格对应的盈亏数组（单位：元）
    """
    # 输入校验
    if any(arg < 0 for arg in [stock_purchase_price, call_premium, strike_price]):
        raise ValueError("价格参数必须为非负数")
    
    # 手续费预计算
    buy_stock_fee = calculate_stock_fee(stock_purchase_price)
    sell_stock_fee = calculate_stock_fee(strike_price)
    sell_call_fee = calculate_option_fee()
    
    # 处理负股价（虽然实际不可能，但增强鲁棒性）
    safe_prices = np.clip(expiration_prices, 0, None)
    
    # 行权场景下的盈亏
    profit_exercised = (
        (strike_price - stock_purchase_price + call_premium) * LOT_SIZE
        - buy_stock_fee - sell_stock_fee - sell_call_fee
    )
    
    # 未行权场景下的盈亏
    profit_not_exercised = (
        (safe_prices - stock_purchase_price + call_premium) * LOT_SIZE
        - buy_stock_fee - sell_call_fee
    )
    
    return np.where(safe_prices > strike_price, profit_exercised, profit_not_exercised)

# --------------------------
# 策略评估模块
# --------------------------
def evaluate_covered_call_strategy(
    current_stock_price: float,
    options_df: pd.DataFrame,
    volatility: float,
    risk_free_rate: float = 0.0  # 添加无风险利率参数
) -> pd.DataFrame:
    """评估备兑开仓策略的关键指标
    
    Args:
        current_stock_price: 标的股票当前价格
        options_df: 包含期权数据的DataFrame，必须包含列：
            ['call_price', 'strike_price', 'days_to_expiration']
        volatility: 标的股票年化波动率
        risk_free_rate: 无风险利率（默认0）
    
    Returns:
        添加评估指标的DataFrame
    
    Raises:
        ValueError: 如果输入DataFrame缺少必要列
    """
    # 输入校验
    required_columns = {'call_price', 'strike_price', 'days_to_expiration'}
    if not required_columns.issubset(options_df.columns):
        missing = required_columns - set(options_df.columns)
        raise ValueError(f"缺少必要列: {missing}")
    
    df = options_df.copy()
    
    # 预计算固定费用
    stock_fee = calculate_stock_fee(current_stock_price)
    option_fee = calculate_option_fee()
    total_fee = stock_fee + option_fee
    
    # 时间相关参数计算
    df['T'] = df['days_to_expiration'] / 365
    df['sigma_T'] = volatility * np.sqrt(df['T'])
    df['mu'] = (
        np.log(current_stock_price) 
        + (risk_free_rate - 0.5 * volatility**2) * df['T']  # 添加无风险利率
    )
    
    # 关键指标计算
    df['premium'] = df['call_price'] * LOT_SIZE - option_fee
    df['break_even_price'] = current_stock_price - df['call_price'] + total_fee / LOT_SIZE
    
    # 概率计算
    def _calc_probability(x: pd.Series, is_break_even: bool = False) -> pd.Series:
        """封装概率计算逻辑"""
        s = x['break_even_price'] if is_break_even else x['strike_price']
        return lognorm.cdf(
            s, x['sigma_T'], 
            scale=np.exp(x['mu'])
        ) * 100
    
    df['expire_worthless_probability (%)'] = df.apply(_calc_probability, axis=1)
    df['break_even_probability (%)'] = 100 - df.apply(
        _calc_probability, axis=1, is_break_even=True
    )
    
    # 预期收益率计算
    def _calculate_expected_profit(row: pd.Series) -> float:
        """计算单个期权的预期收益"""
        # 动态调整价格范围
        upper_bound = np.exp(row['mu'] + 3 * row['sigma_T'])
        price_range = np.linspace(0, upper_bound, NUM_INTEGRATION_POINTS)
        
        # 概率密度函数
        pdf = lognorm.pdf(price_range, row['sigma_T'], scale=np.exp(row['mu']))
        
        # 计算盈亏
        profits = calculate_covered_call_profit(
            price_range, current_stock_price, row['call_price'], row['strike_price']
        )
        
        # 数值积分
        return simps(profits * pdf, price_range)
    
    # 应用向量化计算
    df['expected_profit'] = df.apply(_calculate_expected_profit, axis=1)
    
    # 收益率计算
    cost_basis = (current_stock_price - df['call_price']) * LOT_SIZE + total_fee
    df['expected_profit_rate (%)'] = np.where(
        cost_basis > 0,
        (df['expected_profit'] / cost_basis) * 100,
        0
    )
    
    # 年化收益率
    df['expected_profit_rate_annualized (%)'] = df['expected_profit_rate (%)'] / df['T']
    
    return df[
        ['代码', '名称', 'expiration_date', 'days_to_expiration',
         'call_price', 'strike_price', 'premium', 
         'expire_worthless_probability (%)', 'break_even_probability (%)',
         'expected_profit', 'expected_profit_rate (%)', 
         'expected_profit_rate_annualized (%)']
    ]


import pandas as pd
from utils.data_clean_utils import clean_call_options_data
from utils.implied_volatility import calculate_general_implied_volatility

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