import streamlit as st
import pandas as pd
import numpy as np
import pyotp
from SmartApi.smartConnect import SmartConnect
from datetime import datetime, timedelta, date
import plotly.express as px
import pandas_ta as ta
from scipy.stats import entropy
from io import BytesIO

from Stock_list_token import stock_list


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


# ------------------- EXCEL DOWNLOAD HELPERS -------------------
def df_to_excel_bytes(df: pd.DataFrame, sheet_name="Sheet1") -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return output.getvalue()


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

if "last_scan_df" not in st.session_state:
    st.session_state.last_scan_df = pd.DataFrame()

if "last_watchlist_check" not in st.session_state:
    st.session_state.last_watchlist_check = pd.DataFrame()


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


# ------------------- FETCH CANDLES (DATE RANGE) -------------------
def fetch_candles(symbol: str, token: str, interval="ONE_HOUR",
                  from_date: date = None, to_date: date = None) -> pd.DataFrame:
    """
    Pull OHLCV from Angel One historical candle API using date range.
    Chunking by ~30 days to avoid API issues.
    """
    if from_date is None or to_date is None:
        return pd.DataFrame()

    # Convert dates to datetimes
    start_dt = datetime.combine(from_date, datetime.min.time())
    end_dt = datetime.combine(to_date, datetime.max.time())

    if start_dt >= end_dt:
        return pd.DataFrame()

    chunk_days = 30
    rows = []

    cur_start = start_dt
    while cur_start < end_dt:
        cur_end = min(cur_start + timedelta(days=chunk_days), end_dt)

        params = {
            "exchange": "NSE",
            "symboltoken": str(token),
            "interval": interval,
            "fromdate": cur_start.strftime("%Y-%m-%d %H:%M"),
            "todate": cur_end.strftime("%Y-%m-%d %H:%M"),
        }

        resp = obj.getCandleData(params)
        if resp and resp.get("status") and resp.get("data"):
            rows.extend(resp["data"])

        cur_start = cur_end

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["DateTime", "Open", "High", "Low", "Close", "Volume"])
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df = df.drop_duplicates(subset=["DateTime"]).sort_values("DateTime")
    df.set_index("DateTime", inplace=True)

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna()
    return df


# ------------------- MARKET FOLDING COMPUTATION -------------------
def safe_bbw(close: pd.Series, length=20, std=2.0) -> pd.Series:
    bb = ta.bbands(close, length=length, std=std)
    if bb is None or bb.empty:
        return pd.Series(index=close.index, dtype="float64")

    bbb_cols = [c for c in bb.columns if c.endswith("BBB") or "BBB_" in c]
    if bbb_cols:
        return bb[bbb_cols[0]]

    # fallback
    bbu_cols = [c for c in bb.columns if "BBU" in c]
    bbl_cols = [c for c in bb.columns if "BBL" in c]
    bbm_cols = [c for c in bb.columns if "BBM" in c]
    if bbu_cols and bbl_cols and bbm_cols:
        upper = bb[bbu_cols[0]]
        lower = bb[bbl_cols[0]]
        mid = bb[bbm_cols[0]].replace(0, np.nan)
        return (upper - lower) / mid

    return pd.Series(index=close.index, dtype="float64")


def compute_folding(df: pd.DataFrame, window_size=50, smooth=5) -> pd.DataFrame:
    df = df.copy()

    df["RSI"] = ta.rsi(df["Close"], length=14) / 100.0
    bbw_raw = safe_bbw(df["Close"], length=20, std=2.0)
    df["BBW"] = (bbw_raw - bbw_raw.rolling(50).min()) / (bbw_raw.rolling(50).max() - bbw_raw.rolling(50).min())

    df["NATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=14) / df["Close"]

    vol5 = df["Volume"].rolling(5).mean()
    vol10 = df["Volume"].rolling(10).mean().replace(0, np.nan)
    df["VolOsc"] = (vol5 - vol10) / vol10
    df["VolOsc"] = np.tanh(df["VolOsc"])

    df["CMF"] = ta.cmf(df["High"], df["Low"], df["Close"], df["Volume"], length=20)

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

    # Future Vol (next 24 candles)
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

# Date range selection (replaces lookback days)
default_to = date.today()
default_from = default_to - timedelta(days=180)
from_date = st.sidebar.date_input("From Date", value=default_from)
to_date = st.sidebar.date_input("To Date", value=default_to)

window_size = st.sidebar.slider("Window Size", 20, 200, 50, step=5)
smooth = st.sidebar.slider("Smoothing", 1, 20, 5, step=1)


# ===================== TAB 1: SINGLE STOCK =====================
if tabs == "Single Stock Analyzer":
    st.title("🧬 Single Stock Market Folding Analyzer")

    symbol = st.selectbox("Select Stock", options=sorted(stock_list.keys()))
    token = stock_list[symbol]

    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("➕ Add to Entropy Watchlist"):
            wl = st.session_state.watchlist
            exists = (wl["stock"] == symbol).any()
            if not exists:
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
            st.error("No candle data. Try ONE_DAY interval or change date range.")
            st.stop()

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

        alert = detect_collapse_alert(out, drop_threshold=0.35)
        if alert:
            st.error(f"⚠️ {alert['message']}")
            add_alert(symbol, token, alert["type"], alert["message"], alert["score"])

        st.subheader("📈 Price colored by Folding Score")
        fig = px.scatter(
            usable.reset_index(),
            x="DateTime", y="Close",
            color="Folding_Score",
            title=f"{symbol} | Price vs Folding Score",
        )
        fig.update_traces(marker=dict(size=4))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📉 Folding Score (line)")
        fig2 = px.line(usable.reset_index(), x="DateTime", y="Folding_Score", title="Folding Score Over Time")
        st.plotly_chart(fig2, use_container_width=True)

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
    batch_index = labels.index(batch_choice)
    symbols = chunks[batch_index]

    st.info(f"✅ Selected: {batch_choice} | Stocks in this batch: {len(symbols)}")

    scan_btn = st.button("▶️ Run Selected Batch Scan")
    if scan_btn:
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

                # Status tags
                status = "HEALTHY_COMPLEXITY" if last["Folding_Score"] > usable["Folding_Score"].median() else "LOW_COMPLEXITY"
                if change < -0.35:
                    status = "COLLAPSE_RISK"
                    msg = f"{sym}: Folding score drop detected (Δ {change:.3f})"
                    add_alert(sym, token, "COLLAPSE_RISK", msg, float(last["Folding_Score"]))

                results.append([sym, float(last["Close"]), float(last["Folding_Score"]), change, status])

            except Exception:
                pass

            progress.progress((idx + 1) / n)

        if not results:
            st.warning("No scan results for this batch. Try ONE_DAY interval or change date range.")
        else:
            df_res = pd.DataFrame(results, columns=["Stock", "Last Close", "Last Folding Score", "Score Change", "Status"])
            df_res = df_res.sort_values("Last Folding Score", ascending=True)

            st.session_state.last_scan_df = df_res.copy()

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

                tag = "✅ Healthy" if change >= -0.35 else "⚠️ Collapse Risk"
                rows.append([sym, float(last["Close"]), float(last["Folding_Score"]), change, tag])

                if change < -0.35:
                    msg = f"{sym}: Watchlist collapse risk (Δ {change:.3f})"
                    add_alert(sym, token, "WATCHLIST_COLLAPSE_RISK", msg, float(last["Folding_Score"]))

            if rows:
                df_live = pd.DataFrame(rows, columns=["Stock", "Close", "Folding Score", "Score Change", "Tag"])
                df_live = df_live.sort_values("Folding Score", ascending=True)
                st.session_state.last_watchlist_check = df_live.copy()

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
