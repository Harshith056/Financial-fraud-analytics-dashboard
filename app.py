import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="FraudGuard — Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS — PROFESSIONAL THEME
# ─────────────────────────────────────────────

st.markdown("""
<style>
  /* ── Fonts ── */
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: #0F172A !important;
    border-right: 1px solid #1E293B;
  }
  [data-testid="stSidebar"] * { color: #E2E8F0 !important; }
  [data-testid="stSidebarNav"] { display: none; }

  /* ── Main background ── */
  .stApp { background: #111827 !important; }

  /* ── Remove default padding ── */
  .block-container { padding: 1.5rem 2rem !important; }

  /* ── Metric cards ── */
  .metric-card {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
  }
  .metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 24px rgba(59, 130, 246, 0.10);
  }
  .metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 12px 12px 0 0;
  }
  .metric-card.blue::before  { background: #3B82F6; }
  .metric-card.red::before   { background: #EF4444; }
  .metric-card.amber::before { background: #F59E0B; }
  .metric-card.green::before { background: #10B981; }

  .metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #64748B !important;
    margin-bottom: 0.5rem;
  }
  .metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.9rem;
    font-weight: 600;
    color: #F1F5F9 !important;
    line-height: 1.1;
  }
  .metric-sub {
    font-size: 0.76rem;
    color: #64748B !important;
    margin-top: 0.3rem;
  }
  .metric-icon {
    position: absolute;
    right: 1.4rem; top: 50%;
    transform: translateY(-50%);
    font-size: 2.4rem;
    opacity: 0.06;
  }

  /* ── Section headings ── */
  .section-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #3B82F6;
    margin: 2rem 0 0.8rem;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #334155, transparent);
  }

  /* ── Risk badges ── */
  .risk-high {
    background: rgba(239, 68, 68, 0.10);
    border: 1px solid #EF4444;
    color: #FCA5A5;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    padding: 0.55rem 1.1rem;
    border-radius: 6px;
    display: inline-block;
  }
  .risk-low {
    background: rgba(16, 185, 129, 0.08);
    border: 1px solid #10B981;
    color: #6EE7B7;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    padding: 0.55rem 1.1rem;
    border-radius: 6px;
    display: inline-block;
  }

  /* ── Risk meter ── */
  .risk-meter-track {
    background: #0F172A;
    border: 1px solid #334155;
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
    margin: 0.6rem 0;
  }
  .risk-meter-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.6s cubic-bezier(.4,0,.2,1);
  }

  /* ── Form inputs ── */
  .stSelectbox > div > div,
  .stNumberInput > div > div > input,
  .stSlider { border-radius: 8px !important; }

  /* ── Analyze button ── */
  .stButton > button {
    background: #3B82F6 !important;
    color: #FFFFFF !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.04em !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65rem 2rem !important;
    transition: background 0.2s, transform 0.15s !important;
  }
  .stButton > button:hover {
    background: #2563EB !important;
    transform: scale(1.01) !important;
  }

  /* ── Sidebar nav pills ── */
  [data-testid="stRadio"] label {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 0.5rem 0.85rem;
    border-radius: 8px;
    font-size: 0.84rem;
    font-weight: 400;
    color: #94A3B8 !important;
    cursor: pointer;
    transition: background 0.15s, color 0.15s;
  }
  [data-testid="stRadio"] label:has(input:checked) {
    background: rgba(59, 130, 246, 0.12);
    color: #93C5FD !important;
  }
  [data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {
    font-size: 0.84rem !important;
  }
  [data-testid="stRadio"] input[type="radio"] { display: none !important; }

  /* ── Divider ── */
  hr { border-color: #1E293B !important; }

  /* ── Table ── */
  .dataframe { font-size: 0.82rem !important; }

  /* ── Page header ── */
  .page-header {
    display: flex;
    align-items: baseline;
    gap: 12px;
    margin-bottom: 1.5rem;
  }
  .page-header h1 {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 1.45rem;
    font-weight: 600;
    color: #F1F5F9;
    margin: 0;
    letter-spacing: -0.01em;
  }
  .page-header .badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.58rem;
    background: rgba(59, 130, 246, 0.10);
    border: 1px solid rgba(59, 130, 246, 0.25);
    color: #93C5FD;
    padding: 3px 10px;
    border-radius: 999px;
    letter-spacing: 0.1em;
  }

  /* ── Streamlit metric fallback ── */
  [data-testid="metric-container"] {
    background: #1E293B !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    padding: 1rem !important;
  }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD DATA & MODEL
# ─────────────────────────────────────────────

@st.cache_data
def load_data():
    return pd.read_csv("data/PS_20174392719_1491204439457_log.csv")

@st.cache_resource
def load_model():
    model   = joblib.load("models/paysim_model.pkl")
    encoder = joblib.load("models/type_encoder.pkl")
    return model, encoder

df             = load_data()
model, encoder = load_model()

# ─────────────────────────────────────────────
# PREPARE MODEL DATA
# ─────────────────────────────────────────────

@st.cache_data
def prepare_model_data(_encoder):
    mdf = df[["step","type","amount","isFlaggedFraud","isFraud"]].copy()
    mdf["type"] = _encoder.transform(mdf["type"])
    X   = mdf.drop("isFraud", axis=1)
    y   = mdf["isFraud"]
    return X, y

X, y   = prepare_model_data(encoder)
y_pred = model.predict(X)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding:1.2rem 0 1rem;">
      <div style="font-family:'IBM Plex Sans',sans-serif; font-size:1rem;
                  font-weight:600; color:#F1F5F9; letter-spacing:-0.01em;">
        🛡️ FraudGuard
      </div>
      <div style="font-size:0.7rem; color:#475569; margin-top:3px;
                  font-family:'IBM Plex Mono',monospace; letter-spacing:0.08em;">
        FINANCIAL INTELLIGENCE
      </div>
    </div>
    <hr style="border-color:#1E293B; margin-bottom:1rem;">
    """, unsafe_allow_html=True)

    menu = st.radio(
        "",
        ["📊  Dashboard", "🚨  Fraud Prediction", "📈  Model Performance"],
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    fraud_count = int(df["isFraud"].sum())
    total       = len(df)
    pct         = fraud_count / total * 100

    st.markdown(f"""
    <div style="background:#0F172A; border:1px solid #1E293B;
                border-radius:10px; padding:1rem;
                font-family:'IBM Plex Sans',sans-serif;">
      <div style="font-family:'IBM Plex Mono',monospace; font-size:0.58rem;
                  letter-spacing:0.14em; color:#475569;
                  text-transform:uppercase; margin-bottom:0.7rem;">Summary</div>
      <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
        <span style="font-size:0.78rem; color:#94A3B8;">Total Txns</span>
        <span style="font-family:'IBM Plex Mono',monospace; font-size:0.78rem;
                     color:#E2E8F0;">{total:,}</span>
      </div>
      <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
        <span style="font-size:0.78rem; color:#94A3B8;">Fraud Cases</span>
        <span style="font-family:'IBM Plex Mono',monospace; font-size:0.78rem;
                     color:#FCA5A5;">{fraud_count:,}</span>
      </div>
      <div style="display:flex; justify-content:space-between;">
        <span style="font-size:0.78rem; color:#94A3B8;">Fraud Rate</span>
        <span style="font-family:'IBM Plex Mono',monospace; font-size:0.78rem;
                     color:#FCD34D;">{pct:.4f}%</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PLOTLY PROFESSIONAL THEME
# ─────────────────────────────────────────────

PLOT_BG  = "#1E293B"
PAPER_BG = "#1E293B"
GRID_CLR = "#334155"
FONT_CLR = "#94A3B8"
ACCENT   = "#3B82F6"

def dark_layout(fig, title=""):
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family="IBM Plex Mono", size=12, color="#CBD5E1"),
            x=0,
        ),
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
        font=dict(family="IBM Plex Sans", color=FONT_CLR),
        xaxis=dict(gridcolor=GRID_CLR, linecolor=GRID_CLR, tickcolor=GRID_CLR),
        yaxis=dict(gridcolor=GRID_CLR, linecolor=GRID_CLR, tickcolor=GRID_CLR),
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor=GRID_CLR,
            font=dict(size=12),
        ),
    )
    return fig

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def metric_card(label, value, sub="", color="blue", icon="💠"):
    st.markdown(f"""
    <div class="metric-card {color}">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      {"<div class='metric-sub'>" + sub + "</div>" if sub else ""}
      <div class="metric-icon">{icon}</div>
    </div>
    """, unsafe_allow_html=True)

def section_title(text):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE: DASHBOARD
# ─────────────────────────────────────────────

if "Dashboard" in menu:

    st.markdown("""
    <div class="page-header">
      <h1>Analytics Overview</h1>
      <span class="badge">LIVE</span>
    </div>
    """, unsafe_allow_html=True)

    total_transactions = len(df)
    fraud_cases        = int(df["isFraud"].sum())
    fraud_pct          = fraud_cases / total_transactions * 100
    avg_fraud_amount   = df[df["isFraud"]==1]["amount"].mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total Transactions", f"{total_transactions:,}",
                    "All time", "blue", "💳")
    with c2:
        metric_card("Fraud Cases", f"{fraud_cases:,}",
                    "Confirmed fraud", "red", "🚨")
    with c3:
        metric_card("Fraud Rate", f"{fraud_pct:.4f}%",
                    "of all transactions", "amber", "📊")
    with c4:
        metric_card("Avg Fraud Amount",
                    f"${avg_fraud_amount:,.0f}",
                    "per fraud transaction", "green", "💰")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Fraud by type + Hourly trend ──
    section_title("Transaction Intelligence")
    col_l, col_r = st.columns([1, 1.4])

    with col_l:
        fraud_type = (
            df[df["isFraud"]==1]["type"]
            .value_counts()
            .reset_index()
        )
        fraud_type.columns = ["Type", "Count"]

        fig_pie = px.pie(
            fraud_type,
            names="Type",
            values="Count",
            hole=0.55,
            color_discrete_sequence=["#3B82F6","#EF4444","#F59E0B","#10B981","#8B5CF6"],
        )
        fig_pie.update_traces(
            textposition="outside",
            textfont=dict(family="IBM Plex Sans", size=11, color="#94A3B8"),
            pull=[0.04] * len(fraud_type),
        )
        dark_layout(fig_pie, "FRAUD BY TYPE")
        fig_pie.add_annotation(
            text=f"<b>{fraud_cases:,}</b><br><span style='font-size:10px'>cases</span>",
            x=0.5, y=0.5,
            font=dict(family="IBM Plex Mono", size=15, color="#F1F5F9"),
            showarrow=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_r:
        hourly = (
            df[df["isFraud"]==1]
            .groupby("step")
            .size()
            .reset_index(name="count")
        )
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=hourly["step"],
            y=hourly["count"],
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(59,130,246,0.07)",
            line=dict(color="#3B82F6", width=2),
            name="Fraud Events",
        ))
        dark_layout(fig_line, "FRAUD EVENTS OVER TIME (STEPS)")
        fig_line.update_xaxes(title_text="Step (Hour)", title_font=dict(size=11))
        fig_line.update_yaxes(title_text="Cases",       title_font=dict(size=11))
        st.plotly_chart(fig_line, use_container_width=True)

    # ── Row 3: Amount distribution + Box plot ──
    section_title("Amount Analysis")
    col_a, col_b = st.columns(2)

    with col_a:
        sample = df.sample(min(8000, len(df)), random_state=42)
        fig_hist = px.histogram(
            sample, x="amount", nbins=60,
            color_discrete_sequence=["#2563EB"],
            opacity=0.85,
        )
        fig_hist.update_traces(marker_line_width=0)
        dark_layout(fig_hist, "TRANSACTION AMOUNT DISTRIBUTION")
        fig_hist.update_xaxes(title_text="Amount ($)", title_font=dict(size=11))
        fig_hist.update_yaxes(title_text="Count",     title_font=dict(size=11))
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_b:
        fig_box = go.Figure()
        for label, color, filt in [
            ("Legitimate", "#3B82F6", df[df["isFraud"]==0]),
            ("Fraud",      "#EF4444", df[df["isFraud"]==1]),
        ]:
            sample_filt = filt.sample(min(3000, len(filt)), random_state=1)
            fig_box.add_trace(go.Box(
                y=sample_filt["amount"],
                name=label,
                marker_color=color,
                boxmean=True,
                line=dict(width=1.5),
            ))
        dark_layout(fig_box, "AMOUNT: FRAUD vs LEGITIMATE")
        fig_box.update_yaxes(title_text="Amount ($)", title_font=dict(size=11))
        st.plotly_chart(fig_box, use_container_width=True)

    # ── Stacked bar by type ──
    section_title("Type Breakdown")
    type_stats = (
        df.groupby("type")
        .agg(total=("isFraud","count"), fraud=("isFraud","sum"))
        .reset_index()
    )
    type_stats["legit"] = type_stats["total"] - type_stats["fraud"]

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        name="Legitimate", x=type_stats["type"], y=type_stats["legit"],
        marker_color="#2563EB", marker_line_width=0,
    ))
    fig_bar.add_trace(go.Bar(
        name="Fraud", x=type_stats["type"], y=type_stats["fraud"],
        marker_color="#EF4444", marker_line_width=0,
    ))
    fig_bar.update_layout(barmode="stack")
    dark_layout(fig_bar, "TRANSACTIONS BY TYPE (STACKED)")
    fig_bar.update_xaxes(title_text="Type",  title_font=dict(size=11))
    fig_bar.update_yaxes(title_text="Count", title_font=dict(size=11))
    st.plotly_chart(fig_bar, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE: FRAUD PREDICTION
# ─────────────────────────────────────────────

elif "Prediction" in menu:

    st.markdown("""
    <div class="page-header">
      <h1>Real-Time Detection</h1>
      <span class="badge">ML POWERED</span>
    </div>
    """, unsafe_allow_html=True)

    section_title("Transaction Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        transaction_type = st.selectbox(
            "Transaction Type",
            ["CASH_IN","CASH_OUT","DEBIT","PAYMENT","TRANSFER"],
        )
        amount = st.number_input(
            "Amount ($)",
            min_value=0.0,
            step=100.0,
            format="%.2f",
        )

    with col2:
        step = st.slider("Transaction Hour (Step)", 1, 744, 1,
                         help="Step represents 1 hour of simulation time")
        flagged = st.selectbox(
            "System Flagged?",
            [0, 1],
            format_func=lambda x: "Yes — flagged" if x == 1 else "No — not flagged",
        )

    with col3:
        st.markdown("""
        <div style="background:#0F172A; border:1px solid #1E293B;
                    border-radius:10px; padding:1rem; height:100%;">
          <div style="font-family:'IBM Plex Mono',monospace; font-size:0.6rem;
                      letter-spacing:0.14em; color:#475569; margin-bottom:0.8rem;
                      text-transform:uppercase;">Risk Factors</div>
          <div style="font-size:0.8rem; color:#94A3B8; line-height:2;">
            🔴 &nbsp;TRANSFER / CASH_OUT = high risk<br>
            🟡 &nbsp;Large amounts (&gt; $200K)<br>
            🟡 &nbsp;Off-hours steps (300–600)<br>
            🟢 &nbsp;System-flagged = extra scrutiny<br>
            🟢 &nbsp;PAYMENT / CASH_IN = low risk
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("Analyze Transaction", use_container_width=False)

    if analyze_btn:
        with st.spinner("Running inference…"):
            time.sleep(0.4)

        encoded_type = encoder.transform([transaction_type])[0]
        input_df     = pd.DataFrame([{
            "step": step, "type": encoded_type,
            "amount": amount, "isFlaggedFraud": flagged,
        }])

        prediction  = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1] * 100

        section_title("Risk Analysis")

        r1, r2, r3 = st.columns([1.2, 1, 1])

        with r1:
            if prediction == 1:
                st.markdown(f"""
                <div class="risk-high">⚠ Fraud Detected</div>
                <div style="font-family:'IBM Plex Mono',monospace; font-size:2.2rem;
                            font-weight:600; color:#EF4444; margin-top:0.5rem;">
                  {probability:.1f}%
                </div>
                <div style="font-size:0.75rem; color:#64748B;">Risk Score</div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">✓ Legitimate</div>
                <div style="font-family:'IBM Plex Mono',monospace; font-size:2.2rem;
                            font-weight:600; color:#10B981; margin-top:0.5rem;">
                  {probability:.1f}%
                </div>
                <div style="font-size:0.75rem; color:#64748B;">Fraud Probability</div>
                """, unsafe_allow_html=True)

        with r2:
            fill_color = "#EF4444" if probability > 50 else ("#F59E0B" if probability > 20 else "#10B981")
            st.markdown(f"""
            <div style="background:#1E293B; border:1px solid #334155;
                        border-radius:10px; padding:1.2rem;">
              <div style="font-family:'IBM Plex Mono',monospace; font-size:0.6rem;
                          letter-spacing:0.12em; color:#475569; margin-bottom:0.8rem;
                          text-transform:uppercase;">Risk Meter</div>
              <div class="risk-meter-track">
                <div class="risk-meter-fill"
                     style="width:{min(probability,100):.1f}%;
                            background:{fill_color};"></div>
              </div>
              <div style="display:flex; justify-content:space-between;
                          font-size:0.68rem; color:#475569; margin-top:4px;
                          font-family:'IBM Plex Mono',monospace;">
                <span>0%</span><span>50%</span><span>100%</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with r3:
            metric_card(
                "Transaction",
                f"${amount:,.0f}",
                f"{transaction_type} · Step {step}",
                "blue", "💳",
            )

        st.markdown("<br>", unsafe_allow_html=True)
        section_title("Feature Summary")

        features = {
            "Amount":      min(amount / 500_000 * 100, 100),
            "Type Risk":   80 if transaction_type in ["TRANSFER","CASH_OUT"] else 15,
            "Step Risk":   60 if 300 <= step <= 600 else 20,
            "Flag Signal": 90 if flagged == 1 else 5,
        }

        fig_gauge = go.Figure()
        for feat, val in features.items():
            col_ = "#EF4444" if val > 60 else ("#F59E0B" if val > 30 else "#10B981")
            fig_gauge.add_trace(go.Bar(
                x=[feat], y=[val],
                marker_color=col_,
                marker_line_width=0,
                text=[f"{val:.0f}"],
                textposition="outside",
                textfont=dict(family="IBM Plex Mono", size=11, color="#CBD5E1"),
            ))
        dark_layout(fig_gauge, "FEATURE RISK CONTRIBUTION (%)")
        fig_gauge.update_layout(showlegend=False, height=300)
        fig_gauge.update_yaxes(range=[0, 115])
        st.plotly_chart(fig_gauge, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE: MODEL PERFORMANCE
# ─────────────────────────────────────────────

elif "Performance" in menu:

    st.markdown("""
    <div class="page-header">
      <h1>Model Performance</h1>
      <span class="badge">EVAL</span>
    </div>
    """, unsafe_allow_html=True)

    accuracy  = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall    = recall_score(y, y_pred)
    f1        = f1_score(y, y_pred)

    section_title("Core Metrics")
    m1, m2, m3, m4 = st.columns(4)
    with m1: metric_card("Accuracy",  f"{accuracy:.4f}",  "Overall",    "blue",  "🎯")
    with m2: metric_card("Precision", f"{precision:.4f}", "True pos.",  "green", "✅")
    with m3: metric_card("Recall",    f"{recall:.4f}",    "Sensitivity","amber", "🔍")
    with m4: metric_card("F1 Score",  f"{f1:.4f}",        "Harmonic",  "red",   "⚖️")

    st.markdown("<br>", unsafe_allow_html=True)

    section_title("Confusion Matrix & ROC")
    col_l, col_r = st.columns(2)

    with col_l:
        cm = confusion_matrix(y, y_pred)
        fig_cm = go.Figure(go.Heatmap(
            z=cm,
            x=["Pred: Legit", "Pred: Fraud"],
            y=["Actual: Legit", "Actual: Fraud"],
            colorscale=[
                [0.0, "#0F172A"],
                [0.5, "#1D4ED8"],
                [1.0, "#3B82F6"],
            ],
            text=[[str(v) for v in row] for row in cm],
            texttemplate="<b>%{text}</b>",
            textfont=dict(family="IBM Plex Mono", size=18, color="#F1F5F9"),
            showscale=False,
        ))
        dark_layout(fig_cm, "CONFUSION MATRIX")
        fig_cm.update_layout(height=380)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_r:
        try:
            y_prob      = model.predict_proba(X)[:, 1]
            fpr, tpr, _ = roc_curve(y, y_prob)
            roc_auc     = auc(fpr, tpr)

            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode="lines",
                line=dict(color="#3B82F6", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(59,130,246,0.07)",
                name=f"AUC = {roc_auc:.4f}",
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0,1], y=[0,1],
                mode="lines",
                line=dict(color="#475569", width=1, dash="dash"),
                name="Random",
            ))
            dark_layout(fig_roc, "ROC CURVE")
            fig_roc.update_xaxes(title_text="False Positive Rate", title_font=dict(size=11))
            fig_roc.update_yaxes(title_text="True Positive Rate",  title_font=dict(size=11))
            fig_roc.update_layout(height=380)
            st.plotly_chart(fig_roc, use_container_width=True)
        except Exception:
            st.info("ROC curve unavailable for this model.")

    # ── Metrics radar ──
    section_title("Performance Radar")
    categories = ["Accuracy", "Precision", "Recall", "F1 Score"]
    values     = [accuracy, precision, recall, f1]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(59,130,246,0.10)",
        line=dict(color="#3B82F6", width=2),
        name="Model",
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor=PLOT_BG,
            radialaxis=dict(
                visible=True, range=[0, 1],
                gridcolor=GRID_CLR, linecolor=GRID_CLR,
                tickfont=dict(family="IBM Plex Mono", size=9, color="#475569"),
                tickvals=[0.25, 0.5, 0.75, 1.0],
            ),
            angularaxis=dict(
                gridcolor=GRID_CLR,
                linecolor=GRID_CLR,
                tickfont=dict(family="IBM Plex Sans", size=11, color="#94A3B8"),
            ),
        ),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        showlegend=False,
        height=380,
        margin=dict(l=60, r=60, t=60, b=60),
        title=dict(
            text="METRICS RADAR",
            font=dict(family="IBM Plex Mono", size=12, color="#CBD5E1"),
            x=0,
        ),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Prediction distribution ──
    section_title("Prediction Distribution")
    pred_counts = pd.Series(y_pred).value_counts().reset_index()
    pred_counts.columns = ["Label", "Count"]
    pred_counts["Label"] = pred_counts["Label"].map({0:"Legitimate", 1:"Fraud"})

    fig_pred = px.bar(
        pred_counts, x="Label", y="Count",
        color="Label",
        color_discrete_map={"Legitimate":"#3B82F6", "Fraud":"#EF4444"},
        text="Count",
    )
    fig_pred.update_traces(
        texttemplate="%{text:,}",
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=11),
        marker_line_width=0,
    )
    dark_layout(fig_pred, "PREDICTED CLASS DISTRIBUTION")
    fig_pred.update_layout(showlegend=False)
    st.plotly_chart(fig_pred, use_container_width=True)
