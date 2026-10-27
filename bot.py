"""15-minute BTC/USDT research signals; no order placement."""
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss

BASE = 'https://api.kraken.com/0/public/OHLC'
FEATURES = ['ret1','ret3','ret6','rsi14','ema_gap','ema_slope','vol_ratio','range_pct','body_pct','volatility','hour_sin','hour_cos']


def fetch_candles():
    response = requests.get(BASE, params={'pair':'XBTUSDT','interval':15}, timeout=20)
    response.raise_for_status()
    payload = response.json()
    if payload.get('error'):
        raise RuntimeError('; '.join(payload['error']))
    key = next(k for k in payload['result'] if k != 'last')
    raw = payload['result'][key]
    df = pd.DataFrame(raw, columns=['time','open','high','low','close','vwap','volume','count'])
    for col in ['open','high','low','close','volume']:
        df[col] = pd.to_numeric(df[col])
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df.drop_duplicates('time').sort_values('time').set_index('time')
    # Kraken's last row is the in-progress candle; exclude it unconditionally.
    return df.iloc[:-1].copy()


def add_features(df):
    d = df.copy()
    c = d['close']
    d['ret1'] = c.pct_change()
    d['ret3'] = c.pct_change(3)
    d['ret6'] = c.pct_change(6)
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    d['rsi14'] = 100 * up / (up + down).replace(0,np.nan)
    ema9 = c.ewm(span=9, adjust=False).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()
    d['ema_gap'] = ema9 / ema21 - 1
    d['ema_slope'] = ema9.pct_change(3)
    d['vol_ratio'] = d['volume'] / d['volume'].rolling(20).mean()
    d['range_pct'] = (d['high'] - d['low']) / c
    d['body_pct'] = (d['close'] - d['open']) / d['open']
    d['volatility'] = d['ret1'].rolling(20).std()
    hour = d.index.hour + d.index.minute / 60
    d['hour_sin'] = np.sin(2*np.pi*hour/24)
    d['hour_cos'] = np.cos(2*np.pi*hour/24)
    # Label is direction of NEXT completed candle, not current one.
    d['next_return'] = c.shift(-1) / c - 1
    d['target'] = (d['next_return'] > 0).astype(int)
    return d.replace([np.inf,-np.inf],np.nan)


def model():
    return HistGradientBoostingClassifier(max_iter=75, max_leaf_nodes=7, learning_rate=0.05, l2_regularization=2.0, random_state=42)


def run_analysis(candles, threshold=0.60, min_train=250, test_size=120):
    d = add_features(candles)
    labeled = d.dropna(subset=FEATURES + ['next_return']).copy()
    if len(labeled) < min_train + test_size:
        raise ValueError(f'Need at least {min_train+test_size} labeled candles; got {len(labeled)}')
    test = labeled.iloc[-test_size:].copy()
    train = labeled.iloc[:-test_size].copy()
    # Expanding-window walk-forward. Refitting every candle is intentionally slower,
    # but prevents future data leakage and gives honest chronological predictions.
    probs = []
    for i in range(len(test)):
        history = labeled.iloc[:len(train)+i]
        fitted = model().fit(history[FEATURES], history['target'])
        probs.append(float(fitted.predict_proba(test.iloc[[i]][FEATURES])[0,1]))
    test['p_up'] = probs
    test['signal'] = np.where(test['p_up'] >= threshold,'UP',np.where(test['p_up'] <= 1-threshold,'DOWN','NO TRADE'))
    test['correct'] = np.where(test['signal']=='UP',test['target']==1,np.where(test['signal']=='DOWN',test['target']==0,np.nan))
    # Retrain using all rows for which next candle's outcome is known.
    fitted = model().fit(labeled[FEATURES],labeled['target'])
    latest = d.dropna(subset=FEATURES).iloc[[-1]]
    p = float(fitted.predict_proba(latest[FEATURES])[0,1])
    signal = 'UP' if p >= threshold else ('DOWN' if p <= 1-threshold else 'NO TRADE')
    active = test[test['signal']!='NO TRADE']
    stats = {
        'test_candles':len(test),
        'accuracy_all':float(accuracy_score(test['target'],test['p_up']>=0.5)),
        'balanced_accuracy':float(balanced_accuracy_score(test['target'],test['p_up']>=0.5)),
        'brier':float(brier_score_loss(test['target'],test['p_up'])),
        'signal_count':len(active),
        'signal_accuracy':float(active['correct'].mean()) if len(active) else None,
        'up_baseline':float(test['target'].mean()),
    }
    return latest.index[0],p,signal,test,stats
