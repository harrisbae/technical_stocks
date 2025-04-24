"""
포트폴리오 분석 도구

stocks.txt 파일에서 종목 정보를 읽어와 기술적 분석을 수행합니다.
파일 형식: 종목코드/구매가/수량
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from src.technical_analysis_with_stocks import TechnicalAnalysis

def read_stocks_file(file_path='src/data/stocks.txt'):
    """
    stocks.txt 파일을 읽어 종목 정보를 리스트로 반환합니다.
    
    Returns:
    --------
    list of dict: 종목 정보 리스트
        각 종목 정보는 {'symbol': 종목코드, 'price': 구매가, 'quantity': 수량} 형태
    """
    stocks = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:  # 빈 줄 건너뛰기
                continue
                
            parts = line.split('/')
            if len(parts) == 3:
                symbol = parts[0].strip()
                try:
                    price = float(parts[1].strip())
                    quantity = int(parts[2].strip())
                    stocks.append({
                        'symbol': symbol,
                        'price': price,
                        'quantity': quantity
                    })
                except ValueError:
                    print(f"경고: {line} 처리 중 오류 발생. 숫자 형식이 잘못되었습니다.")
    
    return stocks


def get_stock_data(symbol, period='1y'):
    """
    Yahoo Finance에서 종목 데이터를 가져옵니다.
    
    Parameters:
    -----------
    symbol : str
        종목 코드
    period : str
        데이터 기간 (default: '1y')
        
    Returns:
    --------
    pd.DataFrame or None
        종목 데이터 또는 실패 시 None
    """
    try:
        # Special case for Korean stocks with names instead of ticker symbols
        if symbol == "템퍼스AI":
            symbol = "376300.KS"  # Tempus AI ticker symbol
        # 한국 주식인 경우 '.KS' 확장자 추가
        elif symbol.isdigit() and len(symbol) == 6:
            symbol = f"{symbol}.KS"
            
        print(f"데이터 요청 중: {symbol}")
        stock_data = yf.download(symbol, period=period, progress=False)
        
        if len(stock_data) == 0:
            print(f"경고: {symbol}에 대한 데이터를 찾을 수 없습니다.")
            return None
            
        return stock_data
    except Exception as e:
        print(f"오류: {symbol} 데이터 다운로드 실패 - {str(e)}")
        return None


def analyze_stock(symbol, stock_data, purchase_price=None):
    """
    종목에 대한 기술적 분석을 수행합니다.
    
    Parameters:
    -----------
    symbol : str
        종목 코드
    stock_data : pd.DataFrame
        종목 OHLCV 데이터
    purchase_price : float, optional
        구매 가격
        
    Returns:
    --------
    dict
        분석 결과 정보
    """
    ta = TechnicalAnalysis(stock_data)
    
    # 현재 가격 (가장 최근 종가)
    current_price = stock_data['Close'].iloc[-1]
    
    # 기술 지표 계산
    bollinger = ta.bollinger_bands().iloc[-1]
    rsi = ta.rsi().iloc[-1]
    macd_data = ta.macd().iloc[-1]
    adx_data = ta.adx().iloc[-1]
    cci_data = ta.cci().iloc[-1]  # CCI 추가
    
    # 투자 신호 판단
    market_condition = ta.analyze_market_condition()
    
    # 구매가 대비 수익률
    profit_percentage = None
    if purchase_price:
        profit_percentage = ((current_price - purchase_price) / purchase_price) * 100
    
    # 분석 결과 반환
    return {
        'symbol': symbol,
        'price': current_price,
        'purchase_price': purchase_price,
        'profit_percentage': profit_percentage,
        'bollinger': {
            'upper': bollinger['UB'],
            'middle': bollinger['MB'],
            'lower': bollinger['LB'],
            'width': bollinger['BW'],
            'position': bollinger['BP']
        },
        'rsi': rsi,
        'macd': {
            'macd': macd_data['MACD'],
            'signal': macd_data['Signal'],
            'histogram': macd_data['Histogram']
        },
        'adx': {
            'adx': adx_data['ADX'],
            'plus_di': adx_data['+DI'],
            'minus_di': adx_data['-DI']
        },
        'cci': cci_data,  # CCI 추가
        'market_condition': market_condition
    }


def analyze_portfolio(stocks_data):
    """
    주식 포트폴리오를 분석합니다.
    
    Parameters:
    -----------
    stocks_data : list of dict
        주식 정보가 담긴 리스트
    
    Returns:
    --------
    results : list of dict
        각 종목의 분석 결과
    summary : dict
        포트폴리오 요약 정보
    """
    results = []
    total_investment = 0.0
    total_current_value = 0.0
    
    for stock in stocks_data:
        symbol = stock['symbol']
        purchase_price = stock['price']
        quantity = stock['quantity']
        
        print(f"분석 중: {symbol}")
        
        # 주식 데이터 가져오기
        stock_data = get_stock_data(symbol)
        if stock_data is None or len(stock_data) == 0:
            print(f"경고: {symbol}에 대한 데이터를 가져올 수 없습니다. 건너뜁니다.")
            continue
            
        # 기술적 지표 분석
        ta = TechnicalAnalysis(stock_data)
        
        # 현재 가격 (마지막 종가)
        current_price = stock_data['Close'].iloc[-1]
        
        # Series 객체인 경우 스칼라 값으로 변환
        if isinstance(current_price, pd.Series):
            current_price = float(current_price.iloc[0])
            
        # 투자금액과 현재 가치 계산
        investment = purchase_price * quantity
        current_value = current_price * quantity
        
        # Series 객체인 경우 스칼라 값으로 변환
        if isinstance(investment, pd.Series):
            investment = float(investment.iloc[0])
        if isinstance(current_value, pd.Series):
            current_value = float(current_value.iloc[0])
            
        # 총 합계 누적
        total_investment += float(investment)
        total_current_value += float(current_value)
        
        # 수익률 계산
        profit_percentage = ((current_price / purchase_price) - 1) * 100 if purchase_price else 0
        
        # 기술적 지표 계산
        bollinger = ta.bollinger_bands()
        macd_result = ta.macd()
        rsi_value = ta.rsi()
        cci_value = ta.cci()
        adx_result = ta.adx()
        
        # 시장 상황 분석
        market_condition = ta.analyze_market_condition()
        
        # 결과 저장
        result = {
            'symbol': symbol,
            'purchase_price': purchase_price,
            'price': current_price,
            'profit_percentage': profit_percentage,
            'quantity': quantity,
            'investment': investment,
            'current_value': current_value,
            'bollinger': bollinger,
            'macd': macd_result,
            'rsi': rsi_value,
            'cci': cci_value,
            'adx': adx_result,
            'market_condition': market_condition
        }
        
        results.append(result)
    
    # 포트폴리오 요약 정보
    summary = {
        'total_investment': total_investment,
        'total_current_value': total_current_value,
        'profit_percentage': ((total_current_value / total_investment) - 1) * 100 if total_investment else 0
    }
    
    return results, summary


def display_results(results, summary):
    """
    분석 결과를 화면에 출력합니다.
    
    Parameters:
    -----------
    results : list of dict
        각 종목의 분석 결과
    summary : dict
        포트폴리오 요약 정보
    """
    print("\n" + "="*150)
    print("포트폴리오 분석 결과")
    print("="*150)
    
    # 표 헤더 출력
    headers = ["Ticker", "구매가", "현재가", "수익률(%)", "수익금액", 
               "BOLL위치", "BOLL평가", "MACD", "MACD평가", 
               "RSI", "RSI평가", "CCI", "CCI평가", "ADX", "추세방향", "총평가"]
    
    # 헤더 출력
    header_format = "{:<8} {:<8} {:<8} {:<10} {:<10} {:<10} {:<8} {:<10} {:<8} {:<6} {:<8} {:<10} {:<8} {:<6} {:<8} {:<8}"
    print(header_format.format(*headers))
    print("-"*150)
    
    # 각 종목별 결과 출력
    row_format = "{:<8} {:<8.2f} {:<8.2f} {:<10.2f} {:<10.2f} {:<10.2f} {:<8} {:<10.4f} {:<8} {:<6.1f} {:<8} {:<10.1f} {:<8} {:<6.1f} {:<8} {:<8}"
    
    for r in results:
        symbol = r['symbol']
        purchase_price = r['purchase_price'] or 0
        
        # 현재 가격 - Series 처리
        current_price = r['price']
        if isinstance(current_price, pd.Series):
            current_price = float(current_price.iloc[0])
        
        # 수익률 - Series 처리
        profit_percentage = r['profit_percentage']
        if isinstance(profit_percentage, pd.Series):
            profit_percentage = float(profit_percentage.iloc[0])
        profit_percentage = profit_percentage or 0
        
        # 수익금액
        profit_amount = (current_price - purchase_price) * r['quantity']
        
        # BOLL 관련 지표 - Series 처리
        boll_position = r['bollinger']['position']
        if isinstance(boll_position, pd.Series):
            boll_position = float(boll_position.iloc[0])
            
        if boll_position < 20:
            boll_eval = "매수"
        elif boll_position > 80:
            boll_eval = "매도"
        else:
            boll_eval = "중립"
        
        # MACD 관련 지표 - Series 처리
        macd_value = r['macd']['macd']
        macd_signal = r['macd']['signal']
        
        if isinstance(macd_value, pd.Series):
            macd_value = float(macd_value.iloc[0])
        if isinstance(macd_signal, pd.Series):
            macd_signal = float(macd_signal.iloc[0])
            
        macd_diff = macd_value - macd_signal
        
        if macd_value > 0 and macd_diff > 0:
            macd_eval = "강매수"
        elif macd_value > 0 and macd_diff <= 0:
            macd_eval = "약매수"
        elif macd_value <= 0 and macd_diff > 0:
            macd_eval = "약매도"
        else:
            macd_eval = "강매도"
        
        # RSI 관련 지표 - Series 처리
        rsi_value = r['rsi']
        if isinstance(rsi_value, pd.Series):
            rsi_value = float(rsi_value.iloc[0])
            
        if rsi_value < 30:
            rsi_eval = "과매도"
        elif rsi_value > 70:
            rsi_eval = "과매수"
        else:
            rsi_eval = "중립"
        
        # CCI 관련 지표 - Series 처리
        cci_value = r['cci']
        if isinstance(cci_value, pd.Series):
            cci_value = float(cci_value.iloc[0])
            
        if cci_value < -100:
            cci_eval = "과매도"
        elif cci_value > 100:
            cci_eval = "과매수"
        else:
            cci_eval = "중립"
        
        # ADX 관련 지표 - Series 처리
        adx_value = r['adx']['adx']
        plus_di = r['adx']['plus_di']
        minus_di = r['adx']['minus_di']
        
        if isinstance(adx_value, pd.Series):
            adx_value = float(adx_value.iloc[0])
        if isinstance(plus_di, pd.Series):
            plus_di = float(plus_di.iloc[0])
        if isinstance(minus_di, pd.Series):
            minus_di = float(minus_di.iloc[0])
            
        trend_direction = "상승" if plus_di > minus_di else "하락"
        
        # 출력
        print(row_format.format(
            symbol, purchase_price, current_price, profit_percentage, profit_amount,
            boll_position, boll_eval, macd_diff, macd_eval,
            rsi_value, rsi_eval, cci_value, cci_eval, adx_value, trend_direction, r['market_condition']
        ))
    
    # 구분선
    print("-"*150)
    
    # 포트폴리오 요약 정보 출력
    # 요약 정보 Series 처리
    total_investment = summary['total_investment']
    if isinstance(total_investment, pd.Series):
        total_investment = float(total_investment.iloc[0])
        
    total_current_value = summary['total_current_value']
    if isinstance(total_current_value, pd.Series):
        total_current_value = float(total_current_value.iloc[0])
        
    profit_percentage = summary['profit_percentage']
    if isinstance(profit_percentage, pd.Series):
        profit_percentage = float(profit_percentage.iloc[0])
        
    print(f"\n포트폴리오 요약:")
    print(f"총 투자금액: ${total_investment:.2f}")
    print(f"현재 총 가치: ${total_current_value:.2f}")
    print(f"전체 수익률: {profit_percentage:.2f}%")
    print(f"총 수익금액: ${total_current_value - total_investment:.2f}")
    print("="*150)


def plot_portfolio_allocation(results):
    """
    포트폴리오 구성을 파이 차트로 시각화합니다.
    
    Parameters:
    -----------
    results : list of dict
        각 종목의 분석 결과
    """
    # 현재 가치 기준 데이터 준비
    symbols = [r['symbol'] for r in results]
    values = [r['current_value'] for r in results]
    
    # 너무 작은 종목들은 '기타'로 그룹화
    threshold = sum(values) * 0.02  # 전체의 2% 미만인 종목들을 그룹화
    
    grouped_symbols = []
    grouped_values = []
    others_value = 0
    
    for symbol, value in zip(symbols, values):
        if value >= threshold:
            grouped_symbols.append(symbol)
            grouped_values.append(value)
        else:
            others_value += value
    
    if others_value > 0:
        grouped_symbols.append('기타')
        grouped_values.append(others_value)
    
    # 파이 차트 그리기
    plt.figure(figsize=(10, 8))
    plt.pie(grouped_values, labels=grouped_symbols, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')  # 원형 파이 차트를 위해
    plt.title('포트폴리오 구성')
    plt.tight_layout()
    plt.show()


def plot_performance_comparison(results):
    """
    종목별 수익률 비교를 막대 그래프로 시각화합니다.
    
    Parameters:
    -----------
    results : list of dict
        각 종목의 분석 결과
    """
    # 데이터 준비
    symbols = []
    profits = []
    
    for r in results:
        if r['profit_percentage'] is not None:
            symbols.append(r['symbol'])
            profits.append(r['profit_percentage'])
    
    # 수익률 기준으로 정렬
    sorted_indices = np.argsort(profits)
    sorted_symbols = [symbols[i] for i in sorted_indices]
    sorted_profits = [profits[i] for i in sorted_indices]
    
    # 막대 그래프 색상 설정 (이익은 초록색, 손실은 빨간색)
    colors = ['green' if p >= 0 else 'red' for p in sorted_profits]
    
    # 막대 그래프 그리기
    plt.figure(figsize=(12, 8))
    bars = plt.barh(sorted_symbols, sorted_profits, color=colors)
    
    # 가독성 향상을 위한 설정
    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    plt.grid(axis='x', alpha=0.3)
    plt.title('수익률 비교 (%)')
    plt.xlabel('이익/손실 (%)')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 종목 정보 읽기
    stocks = read_stocks_file()
    
    if not stocks:
        print("오류: 종목 정보를 읽을 수 없거나 파일이 비어 있습니다.")
    else:
        # 포트폴리오 분석
        results, summary = analyze_portfolio(stocks)
        
        # 결과 출력
        display_results(results, summary)
        
        # 포트폴리오 구성 시각화
        plot_portfolio_allocation(results)
        
        # 수익률 비교 시각화
        plot_performance_comparison(results) 