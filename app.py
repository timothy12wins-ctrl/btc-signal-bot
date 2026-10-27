"""Public read-only dashboard for GitHub Actions paper signals."""
import os
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from bot import fetch_candles, run_analysis

st.set_page_config(page_title='BTC 15m Signal Lab', page_icon='📈', layout='wide')
st.title('BTC/USDT • 15-Minute Signal Lab')
st.caption('Paper-only forecasts • Kraken XBT/USDT • No order execution or account access')
st.info('Signals are research estimates, not guaranteed probabilities or financial advice. Scheduling can be delayed by GitHub Actions.')

@st.cache_data(ttl=60)
def candles_live():
    return fetch_candles()

@st.cache_data(ttl=60)
def history_live(url):
    if url:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
    else:
        df = pd.read_csv(Path('data/predictions.csv'))
    if not set(['candle_time_utc','p_up','signal','status','correct']).issubset(df.columns):
        raise ValueError('History file missing expected columns')
    return df

history_url = st.secrets.get('HISTORY_URL', os.getenv('HISTORY_URL', ''))

@st.fragment(run_every='60s')
def main():
    try:
        history = history_live(history_url)
        candles = candles_live()
    except Exception as exc:
        st.error(f'Could not load exchange or prediction history: {exc}')
        return
    st.caption(f'Latest exchange completed candle: {candles.index[-1]:%Y-%m-%d %H:%M} UTC • Refreshes while this page is open')
    if len(history):
        history['candle_time_utc'] = pd.to_datetime(history['candle_time_utc'], utc=True)
        history = history.sort_values('candle_time_utc')
        last = history.iloc[-1]
        resolved = history[history['status'] == 'RESOLVED']
        active = resolved[resolved['signal'].isin(['UP','DOWN'])]
        correct = active['correct'].astype(str).str.lower().eq('true')
        a,b,c,d = st.columns(4)
        a.metric('Last recorded signal', last['signal'])
        b.metric('Model P(UP)', f"{float(last['p_up']):.1%}")
        c.metric('Forward signal accuracy', f'{correct.mean():.1%}' if len(active) else 'Not yet available')
        d.metric('Resolved UP/DOWN signals', len(active))
        st.caption(f"Prediction based on completed candle: {last['candle_time_utc']:%Y-%m-%d %H:%M} UTC. Latest status: {last['status']}. Signals are only logged when the scheduled job runs.")
        st.subheader('Persistent forward paper predictions')
        st.dataframe(history.sort_values('candle_time_utc', ascending=False), use_container_width=True, hide_index=True)
        st.download_button('Download full prediction history CSV', history.to_csv(index=False), file_name='btc_paper_history.csv', mime='text/csv')
        if len(active):
            st.caption('Forward accuracy = correct resolved UP/DOWN signals / all resolved UP/DOWN signals. NO TRADE excluded. No execution costs or P&L calculated.')
    else:
        st.warning('No scheduled predictions recorded yet. Run the GitHub Actions workflow manually once to start the history.')
    fig = go.Figure(data=[go.Candlestick(x=candles.index[-150:], open=candles.open[-150:], high=candles.high[-150:], low=candles.low[-150:], close=candles.close[-150:])])
    fig.update_layout(title='Recent completed 15-minute candles', xaxis_rangeslider_visible=False, height=400)
    st.plotly_chart(fig, use_container_width=True)
    with st.expander('Optional historical walk-forward backtest (may take time)'):
        if st.button('Run 100-candle walk-forward backtest'):
            with st.spinner('Training sequential models...'):
                try:
                    _,_,_,test,stats = run_analysis(candles, test_size=100)
                    st.write(stats)
                    st.dataframe(test[['p_up','signal','next_return','correct']], use_container_width=True)
                except Exception as exc:
                    st.error(str(exc))
            st.caption('Backtest is separate from live forward predictions and is not proof of future profitability.')

main()
