"""Scheduled paper-only forecast writer. No orders or account access."""
import csv
import os
from pathlib import Path
import pandas as pd
from bot import fetch_candles, add_features, model, FEATURES

PATH = Path(os.getenv('HISTORY_PATH', 'data/predictions.csv'))
FIELDS = ['candle_time_utc','created_at_utc','close','p_up','signal','target_candle_utc','actual_close','actual_direction','correct','status']
THRESHOLD = 0.60

def predict(candles):
    d = add_features(candles)
    labeled = d.dropna(subset=FEATURES + ['next_return'])
    latest = d.dropna(subset=FEATURES).iloc[[-1]]
    if len(labeled) < 250 or latest.empty:
        raise RuntimeError('Insufficient completed candles for model training')
    fitted = model().fit(labeled[FEATURES], labeled['target'])
    p = float(fitted.predict_proba(latest[FEATURES])[0, 1])
    t = latest.index[-1]
    return {'candle_time_utc': t.isoformat(), 'created_at_utc': pd.Timestamp.now(tz='UTC').isoformat(),
            'close': str(float(latest.close.iloc[0])), 'p_up': f'{p:.8f}',
            'signal': 'UP' if p >= THRESHOLD else ('DOWN' if p <= 1-THRESHOLD else 'NO TRADE'),
            'target_candle_utc': (t + pd.Timedelta(minutes=15)).isoformat(),
            'actual_close': '', 'actual_direction': '', 'correct': '', 'status': 'PENDING'}

def update_history(candles, path=PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open(newline='') as f:
            rows = list(csv.DictReader(f))
    else:
        rows = []
    by_time = {pd.Timestamp(t).isoformat(): float(v) for t, v in candles.close.items()}
    for row in rows:
        if row['status'] != 'PENDING':
            continue
        target = pd.Timestamp(row['target_candle_utc']).isoformat()
        if target in by_time:
            result = by_time[target]
            prior = float(row['close'])
            direction = 'UP' if result > prior else 'DOWN / FLAT'
            row['actual_close'] = str(result)
            row['actual_direction'] = direction
            row['correct'] = '' if row['signal'] == 'NO TRADE' else str((row['signal'] == 'UP') == (result > prior))
            row['status'] = 'RESOLVED'
    new = predict(candles)
    if not any(r['candle_time_utc'] == new['candle_time_utc'] for r in rows):
        rows.append(new)
    # Keep full audit history, including gaps. No invented predictions for missed intervals.
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Latest closed candle: {new['candle_time_utc']}; total rows: {len(rows)}")
    return rows

if __name__ == '__main__':
    update_history(fetch_candles())
