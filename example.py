"""
주식 기술 분석 도구 사용 예시
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.technical_analysis_with_stocks import TechnicalAnalysis

# 샘플 데이터 생성 (실제 사용 시에는 데이터를 로드하세요)
def generate_sample_data():
    dates = pd.date_range(start='2022-01-01', end='2022-12-31')
    np.random.seed(42)
    
    # 상승 추세 가격 생성
    prices = np.cumsum(np.random.normal(loc=0.1, scale=1, size=len(dates))) + 100
    
    # OHLCV 데이터 생성
    sample_data = pd.DataFrame({
        'Date': dates,
        'Open': prices + np.random.normal(loc=0, scale=0.5, size=len(dates)),
        'High': prices + np.random.normal(loc=1, scale=0.5, size=len(dates)),
        'Low': prices - np.random.normal(loc=1, scale=0.5, size=len(dates)),
        'Close': prices,
        'Volume': np.random.normal(loc=1000000, scale=200000, size=len(dates))
    })
    
    # 데이터 일관성 유지
    sample_data['Open'] = sample_data.apply(lambda x: min(x['Open'], x['High']), axis=1)
    sample_data['Low'] = sample_data.apply(lambda x: min(x['Low'], x['Open'], x['Close']), axis=1)
    sample_data['High'] = sample_data.apply(lambda x: max(x['High'], x['Open'], x['Close']), axis=1)
    
    return sample_data

if __name__ == "__main__":
    # 샘플 데이터 생성
    data = generate_sample_data()
    print("데이터 샘플:")
    print(data.head())
    
    # 기술 분석 객체 생성
    ta = TechnicalAnalysis(data)
    
    # 볼린저 밴드 계산
    bollinger = ta.bollinger_bands()
    print("\n볼린저 밴드 결과:")
    print(bollinger.head())
    
    # RSI 계산
    rsi = ta.rsi()
    print("\nRSI 결과:")
    print(rsi.head())
    
    # MACD 계산
    macd = ta.macd()
    print("\nMACD 결과:")
    print(macd.head())
    
    # 매매 신호 생성
    signals = ta.get_trading_signals()
    print("\n매매 신호 (최근 5일):")
    print(signals[['Close', 'Signal_Text']].tail())
    
    # 시장 상황 분석
    market_condition = ta.analyze_market_condition()
    print(f"\n현재 시장 상황: {market_condition}")
    
    # 차트 표시
    print("\n차트를 표시합니다...")
    ta.plot_indicators() 