# financial_ai_tool.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from scipy.stats import norm

# 风险管理模块 - 计算VaR
def calculate_var(portfolio_returns, confidence_level=0.95):
    mean = np.mean(portfolio_returns)
    std_dev = np.std(portfolio_returns)
    var_value = norm.ppf(1 - confidence_level, mean, std_dev)
    return var_value

# 交易策略模块 - 简单均线策略
def moving_average_strategy(data, short_window=40, long_window=100):
    signals = pd.DataFrame(index=data.index)
    signals['price'] = data['Close']
    signals['short_mavg'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
    signals['long_mavg'] = data['Close'].rolling(window=long_window, min_periods=1).mean()
    signals['signal'] = 0
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1, 0)
    signals['positions'] = signals['signal'].diff()
    return signals

# 机器学习模块 - 随机森林预测市场趋势
def train_market_predictor(data):
    features = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    target = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))
    return model

# 数据加载和回测
def backtest_strategy(data, signals):
    initial_capital = float(100000.0)
    positions = pd.DataFrame(index=signals.index).fillna(0.0)
    positions['Stock'] = 100 * signals['signal']
    portfolio = positions.multiply(data['Close'], axis=0)
    pos_diff = positions.diff()
    portfolio['holdings'] = (positions.multiply(data['Close'], axis=0)).sum(axis=1)
    portfolio['cash'] = initial_capital - (pos_diff.multiply(data['Close'], axis=0)).sum(axis=1).cumsum()
    portfolio['total'] = portfolio['cash'] + portfolio['holdings']
    portfolio['returns'] = portfolio['total'].pct_change()
    return portfolio

# 示例运行
if __name__ == '__main__':
    # 读取数据
    data = pd.read_csv('sample_stock_data.csv', index_col='Date', parse_dates=True)
    
    # 风险管理
    returns = data['Close'].pct_change().dropna()
    var = calculate_var(returns)
    print(f"95% VaR: {var}")

    # 交易策略
    signals = moving_average_strategy(data)
    portfolio = backtest_strategy(data, signals)
    
    # 可视化回测结果
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio['total'], label='Portfolio value')
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

    # 市场预测
    model = train_market_predictor(data)
