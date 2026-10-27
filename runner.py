"""Scheduled paper-only BTC forecasts; never places orders."""
import csv
import os
from pathlib import Path
import pandas as pd
from bot import fetch_candles, add_features, model, FEATURES

PATH = Path(os.getenv('HISTORY_PATH', 'data/predictions.csv'))
FIELDS = ['candle_time_utc', 'created_at_utc', 'close', 'p_up', 'signal', 'target_candle_utc', 'actual_close', 'actual_direction', 'correct', 'status']
THRESHOLD = 0.60
INTERVAL = pd.Timedelta(minutes=15)


def is_on_time(row):
    """Kraken timestamps candles at OPEN; target closes 15m after its timestamp."""
    try:
        created = pd.Timestamp(row['created_at_utc'])
        target_open = pd.Timestamp(row['target_candle_utc'])
        source_open = pd.Timestamp(row['candle_time_utc'])
        return (source_open + INTERVAL <= created < target_open + INTERVAL
                and target_open == source_open + INTERVAL)
    except (KeyError, ValueError, TypeError):
        return False


def predict(candles):
    d = add_features(candles)
    labeled = d.dropna(subset=FEATURES + ['next_return'])
    usable = d.dropna(subset=FEATURES)
    if len(labeled) < 250 or usable.empty:
        raise RuntimeError('Insufficient completed candles for model training')
    latest = usable.iloc[[-1]]
    fitted = model().fit(labeled[FEATURES], labeled['target'])
    p = float(fitted.predict_proba(latest[FEATURES])[0, 1])
    t = latest.index[-1]
    now = pd.Timestamp.now(tz='UTC')
    row = {
        'candle_time_utc': t.isoformat(),
        'created_at_utc': now.isoformat(),
        'close': str(float(latest.close.iloc[0])),
        'p_up': f'{p:.8f}',
        'signal': 'UP' if p >= THRESHOLD else ('DOWN' if p <= 1 - THRESHOLD else 'NO TRADE'),
        'target_candle_utc': (t + INTERVAL).isoformat(),
        'actual_close': '', 'actual_direction': '', 'correct': '', 'status': 'PENDING'
    }
    # Never backfill a prediction after the target candle has closed.
    if not is_on_time(row):
        print(f"Skipping late/stale forecast: source={t.isoformat()}, created={now.isoformat()}")
        return None
    return row


def update_history(candles, path=PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open(newline='') as f:
            rows = list(csv.DictReader(f))
    else:
        rows = []
    # fetch_candles excludes Kraken's unfinished last candle.
    completed_closes = {pd.Timestamp(t).isoformat(): float(v) for t, v in candles.close.items()}
    for row in rows:
        if not is_on_time(row):
            row['status'] = 'INVALID_TIMING'
            row['correct'] = ''
            continue
        if row['status'] != 'PENDING':
            continue
        target = pd.Timestamp(row['target_candle_utc']).isoformat()
        if target in completed_closes:
            result = completed_closes[target]
            prior = float(row['close'])
            row['actual_close'] = str(result)
            row['actual_direction'] = 'UP' if result > prior else 'DOWN / FLAT'
            row['correct'] = '' if row['signal'] == 'NO TRADE' else str((row['signal'] == 'UP') == (result > prior))
            row['status'] = 'RESOLVED'
    new = predict(candles)
    if new and not any(r['candle_time_utc'] == new['candle_time_utc'] for r in rows):
        rows.append(new)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Total rows: {len(rows)}; new valid forecast: {new is not None}')
    return rows


if __name__ == '__main__':
    update_history(fetch_candles())
