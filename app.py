import streamlit as st
import pandas as pd
import numpy as np
import pyotp
from SmartApi.smartConnect import SmartConnect
from datetime import datetime, timedelta, date
import plotly.express as px
from scipy.stats import entropy
from io import BytesIO

from Stock_list_token import stock_list


# ------------------- COLOR SETTINGS -------------------
# Regime colors (interpretable)
REGIME_COLORS = {
    "LOW (Structured)": "#8B0000",   # dark red
    "MID (Transition)": "#FF8C00",   # orange
    "HIGH (Chaotic)":   "#00008B",   # dark blue
}


# ------------------- BATCH HELPERS -------------------
def chunk_list(items, chunk_size=200):
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

def make_batch_labels(chunks):
    labels = []
    start = 1
    for i, ch in enumerate(chunks, start=1):
        end = start + len(ch) - 1
        labels.append(f"Batch {i} ({start}-{end})")
        start = end + 1
    return labels


# ------------------- EXCEL DOWNLOAD HELPERS (FIX TZ DATETIMES) -------------------
def make_excel_safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Excel doesn't support timezone-aware datetimes.
    - converts tz-aware datetime columns to tz-naive
    - converts tz-aware datetime index to tz-naive
    """
    out = df.copy()

    # Fix timezone-aware index
    if isinstance(out.index, pd.DatetimeIndex) and out.index.tz is not None:
        out.index = out.index.tz_convert(None)

    # Fix timezone-aware datetime columns
    for col in out.columns:
        if pd.api.types.is_datetime64tz_dtype(out[col]):
            out[col] = out[col].dt.tz_convert(None)

    return out


def df_to_excel_bytes(df: pd.DataFrame, sheet_name="Sheet1") -> bytes:
    df = make_excel_safe(df)
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return output.getvalue()


# ------------------- INDICATORS (NO pandas_ta dependency) -------------------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def bollinger_bandwidth(close: pd.Series, length: int = 20, std_mult: float = 2.0) -> pd.Series:
    ma = close.rolling(length).mean()
    sd = close.rolling(length).std(ddof=0)
    upper = ma + std_mult * sd
    lower = ma - std_mult * sd
    return (upper - lower) / ma.replace(0, np.nan)


def cmf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    denom = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / denom
    mfv = mfm * volume
    return mfv.rolling(period).sum() / volume.rolling(period).sum().replace(0, np.nan)


# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="🧬 Market Folding (Angel One)", layout="wide")
st.sidebar.title("🧬 Market Folding Navigation")


# ------------------- SESSION STATE INIT -------------------
if "alerts" not in st.session_state:
    st.session_state.alerts = pd.DataFrame(columns=[
        "stock", "token", "alert_type", "message", "folding_score", "created_at"
    ])

if "watchlist" not in st.session_state:
    st.session_state.watchlist = pd.DataFrame(columns=["stock", "token", "added_at"])


# ------------------- ANGEL ONE LOGIN (cached) -------------------
@st.cache_resource
def angel_login():
    # ⚠️ Streamlit Cloud: use st.secrets in production
    api_key = "EKa93pFu"
    client_id = "R59803990"
    password = "1234"
    totp_secret = "5W4MC6MMLANC3UYOAW2QDUIFEU"

    totp = pyotp.TOTP(totp_secret).now()
    obj = SmartConnect(api_key=api_key)
    obj.generateSession(client_id, password, totp)
    return obj

obj = angel_login()


# ------------------- INTERNAL: FETCH RAW CANDLES -------------------
def _fetch_raw(symbol: str, token: str, interval: str, from_date: date, to_date: date) -> pd.DataFrame:
    """
    Raw candle fetch from Angel One.
    ONE_DAY -> fromdate/todate as YYYY-MM-DD (no time)
    Others  -> YYYY-MM-DD HH:MM
    """
    start_dt = datetime.combine(from_date, datetime.min.time())
    end_dt = datetime.combine(to_date, datetime.max.time())
    if start_dt >= end_dt:
        return pd.DataFrame()

    chunk_days = 30
    rows = []
    cur_start = start_dt

    while cur_start < end_dt:
        cur_end = min(cur_start + timedelta(days=chunk_days), end_dt)

        if interval == "ONE_DAY":
            from_str = cur_start.strftime("%Y-%m-%d")
            to_str = cur_end.strftime("%Y-%m-%d")
        else:
            from_str = cur_start.strftime("%Y-%m-%d %H:%M")
            to_str = cur_end.strftime("%Y-%m-%d %H:%M")

        params = {
            "exchange": "NSE",
            "symboltoken": str(token),
            "interval": interval,
            "fromdate": from_str,
            "todate": to_str,
        }

        resp = obj.getCandleData(params)
        if resp and resp.get("status") and resp.get("data"):
            rows.extend(resp["data"])

        cur_start = cur_end

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["DateTime", "Open", "High", "Low", "Close", "Volume"])
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    df = df.dropna(subset=["DateTime"])

    # tz-safe
    try:
        df["DateTime"] = df["DateTime"].dt.tz_localize(None)
    except Exception:
        pass

    # numeric cast
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()

    df = df.sort_values("DateTime")
    df = df.drop_duplicates(subset=["DateTime"], keep="last")
    df.set_index("DateTime", inplace=True)
    return df


# ------------------- RESAMPLE HOURLY -> DAILY (fallback) -------------------
def resample_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Convert intraday OHLCV to daily OHLCV."""
    if df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        return pd.DataFrame()

    daily = df.resample("1D").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()

    return daily


# ------------------- FETCH CANDLES (DATE RANGE + ONE_DAY FIX + FALLBACK) -------------------
def fetch_candles(symbol: str, token: str, interval="ONE_HOUR",
                  from_date: date = None, to_date: date = None) -> pd.DataFrame:
    """
    ✅ ONE_DAY FIX:
    - Use YYYY-MM-DD params for ONE_DAY
    - If ONE_DAY returns too few rows, fallback to ONE_HOUR + resample daily
    """
    if from_date is None or to_date is None:
        return pd.DataFrame()

    raw = _fetch_raw(symbol, token, interval, from_date, to_date)

    if interval == "ONE_DAY":
        expected_days = (to_date - from_date).days + 1
        if raw.empty or len(raw) < min(10, expected_days // 4 + 1):
            hourly = _fetch_raw(symbol, token, "ONE_HOUR", from_date, to_date)
            daily = resample_to_daily(hourly)
            return daily

        tmp = raw.reset_index()
        tmp["Date"] = tmp["DateTime"].dt.date
        tmp = tmp.sort_values("DateTime").drop_duplicates(subset=["Date"], keep="last").drop(columns=["Date"])
        tmp = tmp.sort_values("DateTime").set_index("DateTime")
        return tmp

    return raw


# ------------------- MARKET FOLDING COMPUTATION -------------------
def compute_folding(df: pd.DataFrame, window_size=50, smooth=5) -> pd.DataFrame:
    df = df.copy()

    df["RSI"] = rsi(df["Close"], period=14) / 100.0

    bbw_raw = bollinger_bandwidth(df["Close"], length=20, std_mult=2.0)
    df["BBW"] = (bbw_raw - bbw_raw.rolling(50).min()) / (bbw_raw.rolling(50).max() - bbw_raw.rolling(50).min())

    df["NATR"] = atr(df["High"], df["Low"], df["Close"], period=14) / df["Close"]

    vol5 = df["Volume"].rolling(5).mean()
    vol10 = df["Volume"].rolling(10).mean().replace(0, np.nan)
    df["VolOsc"] = (vol5 - vol10) / vol10
    df["VolOsc"] = np.tanh(df["VolOsc"])

    df["CMF"] = cmf(df["High"], df["Low"], df["Close"], df["Volume"], period=20)

    features = ["RSI", "BBW", "NATR", "VolOsc", "CMF"]
    df = df.dropna(subset=features).copy()

    ent = [np.nan] * len(df)
    feat = df[features]

    for i in range(window_size, len(df)):
        window = feat.iloc[i - window_size:i]
        corr = window.corr().fillna(0)

        eigenvalues, _ = np.linalg.eigh(corr)
        eigenvalues = np.abs(eigenvalues)
        s = eigenvalues.sum()
        if s == 0:
            continue

        p = eigenvalues / s
        ent[i] = entropy(p, base=2)

    df["Market_Entropy"] = ent
    df["Folding_Score"] = df["Market_Entropy"].rolling(smooth).mean()

    df["Future_Vol"] = df["Close"].shift(-24).rolling(24).std()
    return df


# ------------------- ALERT LOGIC -------------------
def detect_collapse_alert(out: pd.DataFrame, drop_threshold=0.35):
    x = out.dropna(subset=["Folding_Score"]).copy()
    if len(x) < 3:
        return None

    last = float(x.iloc[-1]["Folding_Score"])
    prev = float(x.iloc[-2]["Folding_Score"])
    change = last - prev

    if change < -abs(drop_threshold):
        return {
            "type": "DIMENSIONAL_COLLAPSE",
            "message": f"Folding Score dropped sharply: {prev:.3f} → {last:.3f} (Δ {change:.3f})",
            "score": last,
            "change": change
        }
    return None


def add_alert(stock, token, alert_type, message, folding_score):
    new_row = pd.DataFrame([{
        "stock": stock,
        "token": token,
        "alert_type": alert_type,
        "message": message,
        "folding_score": folding_score,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }])
    st.session_state.alerts = pd.concat([new_row, st.session_state.alerts], ignore_index=True)


# ------------------- SIDEBAR NAVIGATION -------------------
tabs = st.sidebar.radio("Select Tab", [
    "Single Stock Analyzer",
    "Scanner",
    "Entropy Watchlist",
    "Alerts / Signals"
])

st.sidebar.divider()
interval = st.sidebar.selectbox("Interval", ["ONE_HOUR", "ONE_DAY", "FIFTEEN_MINUTE", "FIVE_MINUTE"], index=0)

default_to = date.today()
default_from = default_to - timedelta(days=180)
from_date = st.sidebar.date_input("From Date", value=default_from)
to_date = st.sidebar.date_input("To Date", value=default_to)

window_size = st.sidebar.slider("Window Size", 20, 200, 50, step=5)
smooth = st.sidebar.slider("Smoothing", 1, 20, 5, step=1)
collapse_threshold = st.sidebar.slider("Collapse Threshold", 0.05, 1.00, 0.35, 0.05)


# ===================== TAB 1: SINGLE STOCK =====================
if tabs == "Single Stock Analyzer":
    st.title("🧬 Single Stock Market Folding Analyzer")

    symbol = st.selectbox("Select Stock", options=sorted(stock_list.keys()))
    token = stock_list[symbol]

    if st.button("➕ Add to Entropy Watchlist"):
        wl = st.session_state.watchlist
        if not (wl["stock"] == symbol).any():
            new_row = pd.DataFrame([{
                "stock": symbol, "token": token,
                "added_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }])
            st.session_state.watchlist = pd.concat([new_row, wl], ignore_index=True)
            st.success(f"✅ {symbol} added to watchlist")
        else:
            st.info("Already in watchlist.")

    if st.button("▶️ Run Analysis"):
        with st.spinner("Fetching candles..."):
            df = fetch_candles(symbol, token, interval=interval, from_date=from_date, to_date=to_date)

        if df.empty:
            st.error("No candle data returned. Try ONE_HOUR or change date range.")
            st.stop()

        st.caption(f"Rows fetched: {len(df)} | Date range: {df.index.min()} → {df.index.max()}")

        with st.spinner("Computing Folding Score..."):
            out = compute_folding(df, window_size=window_size, smooth=smooth)

        usable = out.dropna(subset=["Folding_Score"])
        if usable.empty:
            st.error("Not enough computed values. Increase date range or reduce window size.")
            st.stop()

        last = usable.iloc[-1]
        prev = usable.iloc[-2] if len(usable) > 1 else last
        change = float(last["Folding_Score"] - prev["Folding_Score"])

        st.metric("Latest Folding Score", f"{last['Folding_Score']:.4f}", delta=f"{change:.4f}")
        st.metric("Latest Close", f"{last['Close']:.2f}")

        alert = detect_collapse_alert(out, drop_threshold=collapse_threshold)
        if alert:
            st.error(f"⚠️ {alert['message']}")
            add_alert(symbol, token, alert["type"], alert["message"], alert["score"])

        # ================= INTERPRETABLE PRICE VS SCORE =================
        st.subheader("📈 Price vs Folding Regimes (Easy Interpretation)")

        plot_df = usable.reset_index().copy()
        plot_df["Score_Change"] = plot_df["Folding_Score"].diff()

        if interval == "ONE_DAY":
            plot_df["X"] = plot_df["DateTime"].dt.date
            x_col = "X"
        else:
            x_col = "DateTime"

        q33 = plot_df["Folding_Score"].quantile(0.33)
        q66 = plot_df["Folding_Score"].quantile(0.66)

        def regime_tag(x):
            if x <= q33:
                return "LOW (Structured)"
            elif x <= q66:
                return "MID (Transition)"
            return "HIGH (Chaotic)"

        plot_df["Regime"] = plot_df["Folding_Score"].apply(regime_tag)

        collapse_df = plot_df[plot_df["Score_Change"] < -collapse_threshold].copy()

        plot_df["Price_Median"] = plot_df["Close"].rolling(10).median()

        fig = px.scatter(
            plot_df,
            x=x_col,
            y="Close",
            color="Regime",
            title=f"{symbol} | Price vs Folding Regimes",
            color_discrete_map=REGIME_COLORS,
        )
        fig.update_traces(marker=dict(size=6, opacity=0.85))

        # rolling trend line
        fig.add_scatter(
            x=plot_df[x_col],
            y=plot_df["Price_Median"],
            mode="lines",
            name="Price (Rolling Median)",
            line=dict(width=2, dash="dot", color="#222222"),
        )

        # collapse markers
        if not collapse_df.empty:
            fig.add_scatter(
                x=collapse_df[x_col],
                y=collapse_df["Close"],
                mode="markers",
                name="⚠️ Collapse Drop",
                marker=dict(color="#FF0000", size=10, symbol="x")
            )

        fig.add_annotation(
            xref="paper", yref="paper", x=0.01, y=0.99, showarrow=False,
            text=f"Regime thresholds: Q33={q33:.3f}, Q66={q66:.3f} | Red=Low, Orange=Mid, Blue=High",
            font=dict(size=12)
        )
        st.plotly_chart(fig, use_container_width=True)

        # regime summary
        latest_regime = plot_df["Regime"].iloc[-1]
        pct_low  = (plot_df["Regime"] == "LOW (Structured)").mean() * 100
        pct_mid  = (plot_df["Regime"] == "MID (Transition)").mean() * 100
        pct_high = (plot_df["Regime"] == "HIGH (Chaotic)").mean() * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latest Regime", latest_regime)
        c2.metric("% Time Low",  f"{pct_low:.1f}%")
        c3.metric("% Time Mid",  f"{pct_mid:.1f}%")
        c4.metric("% Time High", f"{pct_high:.1f}%")

        st.caption(
            "Interpretation: LOW(Structured)=more ordered/mean-reverting; HIGH(Chaotic)=unstable/trending regime; MID=transition. "
            "⚠️ Red X marks sudden score drops (possible regime snap)."
        )

        # ================= SCORE LINE CHART =================
        st.subheader("📉 Folding Score (Dark Blue) + Collapse Drops (Red X)")
        score_df = plot_df.copy()
        collapse_score = score_df[score_df["Score_Change"] < -collapse_threshold].copy()

        fig2 = px.line(score_df, x=x_col, y="Folding_Score", title="Folding Score Over Time")
        fig2.update_traces(line=dict(color="#00008B", width=2))

        if not collapse_score.empty:
            fig2.add_scatter(
                x=collapse_score[x_col],
                y=collapse_score["Folding_Score"],
                mode="markers",
                name="⚠️ Collapse Drop",
                marker=dict(color="#FF0000", size=9, symbol="x")
            )

        st.plotly_chart(fig2, use_container_width=True)

        # ---------------- DOWNLOAD ----------------
        st.subheader("⬇️ Download computed data")
        dl_df = usable.reset_index().rename(columns={"DateTime": "datetime"})
        st.download_button(
            "Download Excel (Single Stock)",
            df_to_excel_bytes(dl_df, sheet_name="single_stock"),
            file_name=f"{symbol}_folding_{interval}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.subheader("📊 Quick View (Last 30 rows)")
        st.dataframe(dl_df.tail(30), use_container_width=True)


# ===================== TAB 2: SCANNER (BATCH) =====================
elif tabs == "Scanner":
    st.title("🔎 Market Folding Scanner (Batch-wise)")

    run_name = st.text_input("Scan Run Name", value=f"Scan_{datetime.now().strftime('%Y%m%d_%H%M')}")

    all_symbols = sorted(stock_list.keys())
    chunks = chunk_list(all_symbols, chunk_size=200)
    labels = make_batch_labels(chunks)

    batch_choice = st.selectbox("Select Batch to Scan", options=labels)
    symbols = chunks[labels.index(batch_choice)]

    st.info(f"✅ Selected: {batch_choice} | Stocks in this batch: {len(symbols)}")

    if st.button("▶️ Run Selected Batch Scan"):
        results = []
        progress = st.progress(0)
        n = len(symbols)

        for idx, sym in enumerate(symbols):
            token = stock_list[sym]
            try:
                df = fetch_candles(sym, token, interval=interval, from_date=from_date, to_date=to_date)
                if df.empty:
                    progress.progress((idx + 1) / n)
                    continue

                out = compute_folding(df, window_size=window_size, smooth=smooth)
                usable = out.dropna(subset=["Folding_Score"])
                if usable.empty:
                    progress.progress((idx + 1) / n)
                    continue

                last = usable.iloc[-1]
                prev = usable.iloc[-2] if len(usable) > 1 else last
                change = float(last["Folding_Score"] - prev["Folding_Score"])

                status = "HEALTHY_COMPLEXITY" if last["Folding_Score"] > usable["Folding_Score"].median() else "LOW_COMPLEXITY"
                if change < -collapse_threshold:
                    status = "COLLAPSE_RISK"
                    msg = f"{sym}: Folding score drop detected (Δ {change:.3f})"
                    add_alert(sym, token, "COLLAPSE_RISK", msg, float(last["Folding_Score"]))

                results.append([sym, float(last["Close"]), float(last["Folding_Score"]), change, status])

            except Exception:
                pass

            progress.progress((idx + 1) / n)

        if not results:
            st.warning("No scan results for this batch. Try ONE_HOUR/ONE_DAY with different date range.")
        else:
            df_res = pd.DataFrame(results, columns=["Stock", "Last Close", "Last Folding Score", "Score Change", "Status"])
            df_res = df_res.sort_values("Last Folding Score", ascending=True)

            st.subheader("✅ Batch Scan Results (sorted by lowest Folding Score first)")
            st.dataframe(df_res, use_container_width=True)

            st.download_button(
                "⬇️ Download Excel (Scan Result)",
                df_to_excel_bytes(df_res, sheet_name="scan"),
                file_name=f"{run_name}_{batch_choice.replace(' ', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


# ===================== TAB 3: WATCHLIST =====================
elif tabs == "Entropy Watchlist":
    st.title("👁️ Entropy Watchlist (Regime Monitor)")

    wl = st.session_state.watchlist
    if wl.empty:
        st.info("Watchlist empty. Add from Single Stock Analyzer.")
    else:
        st.dataframe(wl, use_container_width=True)

        st.divider()
        st.subheader("📡 Live Check (Entropy for Watchlist)")

        if st.button("▶️ Refresh Watchlist Entropy"):
            rows = []
            for _, r in wl.iterrows():
                sym = r["stock"]
                token = r["token"]

                df = fetch_candles(sym, token, interval=interval, from_date=from_date, to_date=to_date)
                if df.empty:
                    continue

                out = compute_folding(df, window_size=window_size, smooth=smooth)
                usable = out.dropna(subset=["Folding_Score"])
                if usable.empty:
                    continue

                last = usable.iloc[-1]
                prev = usable.iloc[-2] if len(usable) > 1 else last
                change = float(last["Folding_Score"] - prev["Folding_Score"])

                tag = "✅ Healthy" if change >= -collapse_threshold else "⚠️ Collapse Risk"
                rows.append([sym, float(last["Close"]), float(last["Folding_Score"]), change, tag])

                if change < -collapse_threshold:
                    msg = f"{sym}: Watchlist collapse risk (Δ {change:.3f})"
                    add_alert(sym, token, "WATCHLIST_COLLAPSE_RISK", msg, float(last["Folding_Score"]))

            if rows:
                df_live = pd.DataFrame(rows, columns=["Stock", "Close", "Folding Score", "Score Change", "Tag"])
                df_live = df_live.sort_values("Folding Score", ascending=True)
                st.dataframe(df_live, use_container_width=True)

                st.download_button(
                    "⬇️ Download Excel (Watchlist Check)",
                    df_to_excel_bytes(df_live, sheet_name="watchlist_check"),
                    file_name=f"watchlist_entropy_{interval}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        st.divider()
        with st.expander("🗑 Remove from Watchlist"):
            rem = st.selectbox("Select Stock", wl["stock"].unique())
            if st.button("Remove"):
                st.session_state.watchlist = wl[wl["stock"] != rem].reset_index(drop=True)
                st.success(f"✅ Removed {rem}")

        st.divider()
        st.download_button(
            "⬇️ Download Watchlist (Excel)",
            df_to_excel_bytes(st.session_state.watchlist, sheet_name="watchlist"),
            file_name="entropy_watchlist.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


# ===================== TAB 4: ALERTS =====================
elif tabs == "Alerts / Signals":
    st.title("🚨 Alerts / Signals Log")

    alerts = st.session_state.alerts
    if alerts.empty:
        st.info("No alerts yet.")
    else:
        st.dataframe(alerts, use_container_width=True)

        st.download_button(
            "⬇️ Download Alerts (Excel)",
            df_to_excel_bytes(alerts, sheet_name="alerts"),
            file_name="entropy_alerts.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        if st.button("🧹 Clear Alerts"):
            st.session_state.alerts = st.session_state.alerts.iloc[0:0]
            st.success("Alerts cleared.")
