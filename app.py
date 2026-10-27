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
        resolved = history[history['status'].astype(str).str.upper().eq('RESOLVED')].copy()
        active = resolved[resolved['signal'].astype(str).str.upper().isin(['UP', 'DOWN'])].copy()
        # Explicitly accept only verified boolean outcomes; exclude blanks/unknowns.
        active['outcome'] = active['correct'].astype(str).str.strip().str.lower()
        active = active[active['outcome'].isin(['true', 'false'])].copy()
        active['win'] = active['outcome'].eq('true').astype(int)
        wins = int(active['win'].sum())
        losses = len(active) - wins
        accuracy = wins / len(active) if len(active) else None
        a, b, c, d = st.columns(4)
        a.metric('Last recorded signal', str(last['signal']))
        b.metric('Model P(UP)', f"{float(last['p_up']):.1%}")
        c.metric('Forward signal accuracy', f'{accuracy:.1%}' if accuracy is not None else 'Not yet available')
        d.metric('Scored UP/DOWN signals', len(active))
        st.caption(f"Prediction based on completed candle: {last['candle_time_utc']:%Y-%m-%d %H:%M} UTC. Latest status: {last['status']}. Signals are only logged when the scheduled job runs.")

        st.subheader('Forward signal scorecard')
        w, l, r, pending = st.columns(4)
        w.metric('Wins (correct direction)', wins)
        l.metric('Losses (incorrect direction)', losses)
        recent = active.tail(20)
        r.metric('Last 20 scored accuracy', f"{recent['win'].mean():.1%}" if len(recent) else 'N/A',
                 help='Uses up to 20 most recent scored UP/DOWN predictions; not necessarily 20 consecutive candles.')
        pending.metric('Pending signals', int(history['status'].astype(str).str.upper().eq('PENDING').sum()))
        if len(active):
            breakdown = active.groupby('signal')['win'].agg(['count', 'sum']).reindex(['UP', 'DOWN']).fillna(0)
            breakdown['losses'] = breakdown['count'] - breakdown['sum']
            breakdown['accuracy'] = breakdown.apply(lambda x: x['sum'] / x['count'] if x['count'] else float('nan'), axis=1)
            breakdown = breakdown.rename(columns={'count': 'Signals', 'sum': 'Wins', 'losses': 'Losses', 'accuracy': 'Accuracy'})
            st.write('**UP versus DOWN results**')
            st.dataframe(breakdown, use_container_width=True, column_config={'Accuracy': st.column_config.NumberColumn(format='%.1f%%')})
            # Each record corresponds to one completed, previously recorded signal.
            chart = active[['candle_time_utc', 'win']].copy()
            chart['cumulative_accuracy'] = chart['win'].expanding().mean() * 100
            chart['rolling_20_accuracy'] = chart['win'].rolling(20, min_periods=1).mean() * 100
            chart = chart.set_index('candle_time_utc')[['cumulative_accuracy', 'rolling_20_accuracy']]
            st.write('**Accuracy trend (% of scored signals)**')
            st.line_chart(chart, y_label='Accuracy (%)')
            st.caption('Rolling accuracy uses up to 20 scored signals. NO TRADE, pending and unknown outcomes are excluded. This measures direction accuracy, not trade returns.')
        else:
            st.info('No scored UP/DOWN predictions yet. As future candles close and scheduled jobs resolve signals, accuracy will appear automatically.')
        st.subheader('Persistent forward paper predictions')
        st.dataframe(history.sort_values('candle_time_utc', ascending=False), use_container_width=True, hide_index=True)
        st.download_button('Download full prediction history CSV', history.to_csv(index=False), file_name='btc_paper_history.csv', mime='text/csv')
        st.caption('Accuracy = correct resolved UP/DOWN signals / all scored resolved UP/DOWN signals. NO TRADE excluded. No execution costs or P&L calculated.')
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
