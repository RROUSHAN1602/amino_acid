import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import pyotp
from SmartApi.smartConnect import SmartConnect
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

import pandas_ta as ta
from scipy.stats import entropy

from Stock_list_token import stock_list
def chunk_list(items, chunk_size=200):
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

def make_batch_labels(chunks):
    # returns ["Batch 1 (1-200)", "Batch 2 (201-400)", ...]
    labels = []
    start = 1
    for i, ch in enumerate(chunks, start=1):
        end = start + len(ch) - 1
        labels.append(f"Batch {i} ({start}-{end})")
        start = end + 1
    return labels


# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="🧬 Market Folding (Angel One)", layout="wide")
st.sidebar.title("🧬 Market Folding Navigation")


# ------------------- DB CONNECTION & INIT -------------------
def get_connection():
    return sqlite3.connect("market_folding.db", check_same_thread=False)

def init_db():
    con = get_connection()
    cur = con.cursor()

    # Entropy Watchlist
    cur.execute("""
        CREATE TABLE IF NOT EXISTS entropy_watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stock TEXT NOT NULL,
            token TEXT NOT NULL,
            added_at TEXT NOT NULL
        )
    """)

    # Alerts table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS entropy_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stock TEXT NOT NULL,
            token TEXT NOT NULL,
            alert_type TEXT NOT NULL,
            message TEXT NOT NULL,
            folding_score REAL,
            created_at TEXT NOT NULL
        )
    """)

    # Scan results table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS scan_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_name TEXT NOT NULL,
            interval TEXT NOT NULL,
            lookback_days INTEGER NOT NULL,
            window_size INTEGER NOT NULL,
            smooth INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    # Scan rows
    cur.execute("""
        CREATE TABLE IF NOT EXISTS scan_rows (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_id INTEGER NOT NULL,
            stock TEXT NOT NULL,
            token TEXT NOT NULL,
            last_close REAL,
            last_folding_score REAL,
            score_change REAL,
            status TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(scan_id) REFERENCES scan_results(id)
        )
    """)

    con.commit()
    con.close()

init_db()


# ------------------- ANGEL ONE LOGIN (cached) -------------------
@st.cache_resource
def angel_login():
    # ⚠️ Use st.secrets in production
    api_key = "EKa93pFu"
    client_id = "R59803990"
    password = "1234"
    totp_secret = "5W4MC6MMLANC3UYOAW2QDUIFEU"

    totp = pyotp.TOTP(totp_secret).now()
    obj = SmartConnect(api_key=api_key)
    obj.generateSession(client_id, password, totp)
    return obj

obj = angel_login()


# ------------------- FETCH CANDLES (Angel One) -------------------
def fetch_candles(symbol: str, token: str, interval="ONE_HOUR", lookback_days=180) -> pd.DataFrame:
    """
    Pull OHLCV from Angel One historical candle API.
    Uses chunking to reduce failures.
    """
    end = datetime.now()
    start = end - timedelta(days=lookback_days)

    # chunk size (days)
    chunk_days = 30
    rows = []

    cur_start = start
    while cur_start < end:
        cur_end = min(cur_start + timedelta(days=chunk_days), end)

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

    # numeric cast
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna()
    return df


# ------------------- MARKET FOLDING COMPUTATION -------------------
def safe_bbw(close: pd.Series, length=20, std=2.0) -> pd.Series:
    bb = ta.bbands(close, length=length, std=std)
    if bb is None or bb.empty:
        return pd.Series(index=close.index, dtype="float64")

    # try BBB column
    bbb_cols = [c for c in bb.columns if c.endswith("BBB") or "BBB_" in c]
    if bbb_cols:
        return bb[bbb_cols[0]]

    # fallback (upper-lower)/mid
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

    # Features
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
        window = feat.iloc[i-window_size:i]
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
    """
    Simple collapse: folding_score drops sharply from previous window.
    """
    x = out.dropna(subset=["Folding_Score"]).copy()
    if len(x) < 3:
        return None

    last = x.iloc[-1]["Folding_Score"]
    prev = x.iloc[-2]["Folding_Score"]
    change = last - prev

    if change < -abs(drop_threshold):
        return {
            "type": "DIMENSIONAL_COLLAPSE",
            "message": f"Folding Score dropped sharply: {prev:.3f} → {last:.3f} (Δ {change:.3f})",
            "score": float(last),
            "change": float(change)
        }
    return None


# ------------------- DB HELPERS -------------------
def add_to_watchlist(symbol, token):
    con = get_connection()
    cur = con.cursor()
    cur.execute("INSERT INTO entropy_watchlist (stock, token, added_at) VALUES (?,?,?)",
                (symbol, token, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    con.commit()
    con.close()

def get_watchlist():
    con = get_connection()
    df = pd.read_sql("SELECT * FROM entropy_watchlist ORDER BY added_at DESC", con)
    con.close()
    return df

def remove_from_watchlist(symbol):
    con = get_connection()
    cur = con.cursor()
    cur.execute("DELETE FROM entropy_watchlist WHERE stock=?", (symbol,))
    con.commit()
    con.close()

def log_alert(symbol, token, alert_type, message, folding_score=None):
    con = get_connection()
    cur = con.cursor()
    cur.execute("""
        INSERT INTO entropy_alerts (stock, token, alert_type, message, folding_score, created_at)
        VALUES (?,?,?,?,?,?)
    """, (symbol, token, alert_type, message, folding_score, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    con.commit()
    con.close()

def get_alerts():
    con = get_connection()
    df = pd.read_sql("SELECT * FROM entropy_alerts ORDER BY created_at DESC", con)
    con.close()
    return df

def create_scan(run_name, interval, lookback_days, window_size, smooth):
    con = get_connection()
    cur = con.cursor()
    cur.execute("""
        INSERT INTO scan_results (run_name, interval, lookback_days, window_size, smooth, created_at)
        VALUES (?,?,?,?,?,?)
    """, (run_name, interval, lookback_days, window_size, smooth, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    scan_id = cur.lastrowid
    con.commit()
    con.close()
    return scan_id

def add_scan_row(scan_id, symbol, token, last_close, last_score, score_change, status):
    con = get_connection()
    cur = con.cursor()
    cur.execute("""
        INSERT INTO scan_rows (scan_id, stock, token, last_close, last_folding_score, score_change, status, created_at)
        VALUES (?,?,?,?,?,?,?,?)
    """, (scan_id, symbol, token, last_close, last_score, score_change, status,
          datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    con.commit()
    con.close()

def get_scans():
    con = get_connection()
    df = pd.read_sql("SELECT * FROM scan_results ORDER BY created_at DESC", con)
    con.close()
    return df

def get_scan_rows(scan_id):
    con = get_connection()
    df = pd.read_sql("SELECT * FROM scan_rows WHERE scan_id=? ORDER BY last_folding_score ASC", con, params=(scan_id,))
    con.close()
    return df


# ------------------- SIDEBAR NAVIGATION -------------------
tabs = st.sidebar.radio("Select Tab", [
    "Single Stock Analyzer",
    "Scanner",
    "Entropy Watchlist",
    "Alerts / Signals",
    "Saved Reports"
])

# Common controls in sidebar
st.sidebar.divider()
interval = st.sidebar.selectbox("Interval", ["ONE_HOUR", "ONE_DAY", "FIFTEEN_MINUTE", "FIVE_MINUTE"], index=0)
lookback_days = st.sidebar.slider("Lookback Days", 30, 730, 180, step=30)
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
            add_to_watchlist(symbol, token)
            st.success(f"✅ {symbol} added to Entropy Watchlist")

    if st.button("▶️ Run Analysis"):
        with st.spinner("Fetching candles..."):
            df = fetch_candles(symbol, token, interval=interval, lookback_days=lookback_days)

        if df.empty:
            st.error("No candle data. Try ONE_DAY interval or shorter lookback.")
            st.stop()

        with st.spinner("Computing Folding Score..."):
            out = compute_folding(df, window_size=window_size, smooth=smooth)

        usable = out.dropna(subset=["Folding_Score"])
        if usable.empty:
            st.error("Not enough computed values. Increase lookback or reduce window size.")
            st.stop()

        last = usable.iloc[-1]
        prev = usable.iloc[-2] if len(usable) > 1 else last
        change = float(last["Folding_Score"] - prev["Folding_Score"])

        st.metric("Latest Folding Score", f"{last['Folding_Score']:.4f}", delta=f"{change:.4f}")
        st.metric("Latest Close", f"{last['Close']:.2f}")

        # Alert check
        alert = detect_collapse_alert(out, drop_threshold=0.35)
        if alert:
            st.error(f"⚠️ {alert['message']}")
            log_alert(symbol, token, alert["type"], alert["message"], alert["score"])

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

        st.subheader("📊 Quick View (Last 30 rows)")
        st.dataframe(usable.tail(30), use_container_width=True)


# ===================== TAB 2: SCANNER =====================
elif tabs == "Scanner":
    st.title("🔎 Market Folding Scanner (Batch-wise)")

    run_name = st.text_input("Scan Run Name", value=f"Scan_{datetime.now().strftime('%Y%m%d_%H%M')}")

    # --- Create batches of 200 ---
    all_symbols = sorted(stock_list.keys())
    chunks = chunk_list(all_symbols, chunk_size=200)
    labels = make_batch_labels(chunks)

    # Batch selector
    batch_choice = st.selectbox("Select Batch to Scan", options=labels)
    batch_index = labels.index(batch_choice)
    symbols = chunks[batch_index]

    st.info(f"✅ Selected: {batch_choice} | Stocks in this batch: {len(symbols)}")
    with st.expander("📋 View symbols in this batch"):
        st.write(symbols)

    scan_btn = st.button("▶️ Run Selected Batch Scan")

    if scan_btn:
        scan_id = create_scan(run_name, interval, lookback_days, window_size, smooth)
        st.success(f"✅ Scan Created: ID {scan_id}")

        results = []
        progress = st.progress(0)
        n = len(symbols)

        for idx, sym in enumerate(symbols):
            token = stock_list[sym]
            try:
                df = fetch_candles(sym, token, interval=interval, lookback_days=lookback_days)
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

                add_scan_row(
                    scan_id=scan_id,
                    symbol=sym,
                    token=token,
                    last_close=float(last["Close"]),
                    last_score=float(last["Folding_Score"]),
                    score_change=change,
                    status=status
                )

                results.append([sym, float(last["Close"]), float(last["Folding_Score"]), change, status])

                # auto-alert
                if status == "COLLAPSE_RISK":
                    msg = f"{sym}: Folding score drop detected (Δ {change:.3f})"
                    log_alert(sym, token, "COLLAPSE_RISK", msg, float(last["Folding_Score"]))

            except Exception:
                # Keep scanning even if one stock fails
                pass

            progress.progress((idx + 1) / n)

        if not results:
            st.warning("No scan results for this batch. Try ONE_DAY interval or reduce window size.")
        else:
            df_res = pd.DataFrame(results, columns=["Stock", "Last Close", "Last Folding Score", "Score Change", "Status"])

            st.subheader("✅ Batch Scan Results (sorted by lowest Folding Score first)")
            df_res = df_res.sort_values("Last Folding Score", ascending=True)
            st.dataframe(df_res, use_container_width=True)

            st.download_button(
                "⬇️ Download CSV",
                df_res.to_csv(index=False).encode("utf-8"),
                file_name=f"{run_name}_{batch_choice.replace(' ', '_')}.csv",
                mime="text/csv"
            )


# ===================== TAB 3: WATCHLIST =====================
elif tabs == "Entropy Watchlist":
    st.title("👁️ Entropy Watchlist (Regime Monitor)")

    df_w = get_watchlist()
    if df_w.empty:
        st.info("Watchlist empty. Add from Single Stock Analyzer.")
    else:
        st.dataframe(df_w, use_container_width=True)

        st.divider()
        st.subheader("📡 Live Check (Entropy for Watchlist)")

        if st.button("▶️ Refresh Watchlist Entropy"):
            rows = []
            for _, r in df_w.iterrows():
                sym = r["stock"]
                token = r["token"]

                df = fetch_candles(sym, token, interval=interval, lookback_days=lookback_days)
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
                    log_alert(sym, token, "WATCHLIST_COLLAPSE_RISK", msg, float(last["Folding_Score"]))

            if rows:
                df_live = pd.DataFrame(rows, columns=["Stock", "Close", "Folding Score", "Score Change", "Tag"])
                st.dataframe(df_live.sort_values("Folding Score", ascending=True), use_container_width=True)

        st.divider()
        with st.expander("🗑 Remove from Watchlist"):
            rem = st.selectbox("Select Stock", df_w["stock"].unique())
            if st.button("Remove"):
                remove_from_watchlist(rem)
                st.success(f"✅ Removed {rem}")


# ===================== TAB 4: ALERTS =====================
elif tabs == "Alerts / Signals":
    st.title("🚨 Alerts / Signals Log")

    df_a = get_alerts()
    if df_a.empty:
        st.info("No alerts yet.")
    else:
        st.dataframe(df_a, use_container_width=True)


# ===================== TAB 5: SAVED REPORTS =====================
elif tabs == "Saved Reports":
    st.title("📁 Saved Scan Reports")

    scans = get_scans()
    if scans.empty:
        st.info("No scans saved yet. Run Scanner tab first.")
    else:
        st.dataframe(scans, use_container_width=True)

        scan_id = st.selectbox("Select Scan ID to view", options=scans["id"].tolist())
        rows = get_scan_rows(scan_id)

        st.subheader(f"Scan Rows for ID: {scan_id}")
        st.dataframe(rows, use_container_width=True)

        st.download_button(
            "⬇️ Download Scan Rows CSV",
            rows.to_csv(index=False).encode("utf-8"),
            file_name=f"scan_{scan_id}_rows.csv",
            mime="text/csv"
        )
