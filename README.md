# BTC/USDT 15-minute signal dashboard (paper only)

This project uses **Kraken XBT/USDT**, not BTC/USD. It forecasts the direction of the next *completed* 15-minute candle using a gradient boosting classifier trained on past closed candles. It records forecasts before the target candle closes and resolves them afterward. No trades, orders, wallet connections, or deposits.

## 1. Upload to GitHub

1. Create a **public GitHub repository** (for example `btc-signal-lab`).
2. Upload **all files and folders** from this extracted project to the repository's `main` branch, including `.github/workflows/predict.yml`, `.streamlit/config.toml`, and `data/predictions.csv`. GitHub's web uploader can omit hidden folders; use Git or GitHub Desktop if necessary.
3. Under **Settings → Actions → General → Workflow permissions**, enable **Read and write permissions** for `GITHUB_TOKEN`. If organization rules prohibit this, a repository admin must adjust them. Scheduled workflows run only on the default branch.
4. Open **Actions → Record BTC 15-minute paper signals → Run workflow**. Check the run logs. It should commit a new row to `data/predictions.csv`. The schedule runs at minutes 03, 18, 33, and 48 UTC; GitHub may delay or skip runs, especially for public/free services. There is no uptime guarantee. GitHub may automatically disable scheduled workflows after prolonged repository inactivity.

## 2. Deploy the mobile-accessible Streamlit website

1. Sign in to https://share.streamlit.io/ with GitHub and choose **Create app → Deploy a public app from GitHub**.
2. Select your repository and `main` branch. Set **Main file path** to `app.py`.
3. In **Advanced settings → Secrets**, paste the following TOML, replacing YOUR_NAME and YOUR_REPO with your actual GitHub user and repository:

   ```toml
   HISTORY_URL = "https://raw.githubusercontent.com/YOUR_NAME/YOUR_REPO/main/data/predictions.csv"
   ```

   This is a public URL, not a secret. It is placed in Streamlit settings for convenience; never put passwords or exchange keys in this project.
4. Deploy. Bookmark the `*.streamlit.app` link on your phone. The page refreshes every 60 seconds **while open**; the GitHub workflow continues running when your computer is off.

## How accuracy works

- One row is written per distinct completed source candle (duplicates suppressed).
- Each row forecasts the next 15-minute candle, not the candle used for training.
- On later runs, forecasts are marked `RESOLVED` if their target candle is still in the Kraken OHLC response. Kraken returns a limited rolling window, so very old missed resolutions may remain `PENDING`.
- `NO TRADE` is excluded from directional accuracy. Flat prices count as `DOWN / FLAT`.
- **Missing scheduler intervals are not backfilled** as if they were real-time forecasts. The recorded timestamp reveals delays.
- The ML probability is uncalibrated. Historical backtests and forward accuracy do not imply profitable trading; fees, spread and slippage are not modeled.
- This is **not** an exchange trading bot. It makes no trades.

## Local testing

Requires Python 3.11+:

```bash
pip install -r requirements.txt
python runner.py
streamlit run app.py
```

Without `HISTORY_URL`, the dashboard reads local `data/predictions.csv`. Run `python runner.py` again after a new 15-minute candle completes to record another prediction and resolve previous predictions. For the hosted dashboard, the Streamlit secret must point to the GitHub raw CSV.

## Cost / limitations

GitHub Actions and Streamlit Community Cloud can be free within their usage limits, but services may throttle, sleep, change limits, or be unavailable. Public repo means the prediction CSV is public. Kraken's public endpoint can rate-limit or be unavailable. This is a prototype, not a guaranteed always-on service; use a dedicated scheduler/database for stronger reliability.
