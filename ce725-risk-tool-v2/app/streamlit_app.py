"""Construction schedule and cost risk tool. Run from the repo root:
    streamlit run app/streamlit_app.py
"""
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from charts import gantt_chart, gantt_frame, ladder_chart
from engine import NetworkError, activity_table, baseline_comparison, build_network, percentiles, run_simulation
from engine.analysis import (contingency_summary, evaluate_mitigation, model_ladder, risk_contributions)
from engine.risks import validate_risks
from finance.cashflow import deterministic_finance, run_finance

DATA = ROOT / "data"
st.set_page_config(page_title="Construction schedule and cost risk", layout="wide")


@st.cache_data
def read_csv(path):
    return pd.read_csv(path)


@st.cache_data(show_spinner="Running simulation...")
def simulate(acts, risks, rho, n_iter, seed, dist, inputs):
    net = build_network(acts)
    res = run_simulation(net, n_iter=n_iter, seed=seed, distribution=dist, rho_by_group=rho, risks=risks,
                         overhead_per_month_lakh=inputs["overhead_per_month_lakh"])
    fin, flows = run_finance(res, inputs)
    return net, res, fin, flows


@st.cache_data(show_spinner="Building the model ladder...")
def cached_ladder(acts, rho, risks, overhead, n_iter, seed):
    return model_ladder(build_network(acts), rho, risks, overhead, n_iter, seed)


@st.cache_data(show_spinner="Testing each risk...")
def cached_contrib(acts, rho, risks, overhead, n_iter, seed):
    return risk_contributions(build_network(acts), rho, risks, overhead, n_iter, seed)


@st.cache_data(show_spinner="Comparing with and without the fix...")
def cached_mitigation(acts, rho, risks, rid, p, scale, cost, inputs, n_iter, seed):
    return evaluate_mitigation(build_network(acts), rho, risks, rid, p, scale, cost, inputs, n_iter, seed)


# ---------------- Sidebar ----------------
st.sidebar.header("Project data")
dataset = st.sidebar.radio("Activities", ["Case study (G+10 residential)", "Upload CSV"])
if dataset.startswith("Case"):
    acts = read_csv(DATA / "case_study_activities.csv")
else:
    up = st.sidebar.file_uploader("Activity CSV (same columns as the case study file)", type="csv")
    if up is None:
        st.info("Upload an activity CSV, or choose the case study.")
        st.stop()
    acts = pd.read_csv(up)
start_date = st.sidebar.date_input("Construction start date", dt.date(2027, 1, 1))

st.sidebar.header("Simulation")
n_iter = st.sidebar.select_slider("Iterations", [1_000, 5_000, 10_000, 20_000], value=10_000)
seed = int(st.sidebar.number_input("Random seed (same seed gives same results)", value=42, step=1))
dist = st.sidebar.radio("Distribution", ["pert", "triangular"], format_func=str.capitalize, horizontal=True)
use_corr = st.sidebar.toggle("Activities in the same group slip together", value=True)
use_risks = st.sidebar.toggle("Include risk events", value=True)

base_inputs = json.loads((DATA / "finance_inputs.json").read_text())
with st.sidebar.expander("Finance inputs"):
    inputs = {}
    fields = [("land_cost_lakh", "Land cost (₹ lakh)", 1.0), ("approvals_cost_lakh", "Approvals (₹ lakh)", 1.0),
              ("saleable_area_sqft", "Saleable area (sq ft)", 1.0), ("sale_price_per_sqft_inr", "Sale price (₹ per sq ft)", 1.0),
              ("presale_share", "Share sold before completion", 0.05), ("months_to_sell_after_completion", "Months to sell the rest", 1.0),
              ("brokerage_pct", "Brokerage (share of sales)", 0.005), ("debt_share", "Loan share of funding gap", 0.05),
              ("interest_rate_annual", "Loan interest rate (per year)", 0.005), ("discount_rate_annual", "Discount rate, project (per year)", 0.005),
              ("equity_discount_rate_annual", "Cost of equity (per year)", 0.005), ("overhead_per_month_lakh", "Site overheads (₹ lakh per month)", 0.5)]
    for key, label, step in fields:
        inputs[key] = float(st.number_input(label, value=float(base_inputs[key]), step=step, key=key))
    inputs["months_to_sell_after_completion"] = int(inputs["months_to_sell_after_completion"])

rho = dict(read_csv(DATA / "correlation_groups.csv")[["group", "rho"]].values) if use_corr else None
risks = read_csv(DATA / "risk_register.csv") if use_risks else None

# ---------------- Run ----------------
try:
    net, res, fin, flows = simulate(acts, risks, rho, n_iter, seed, dist, inputs)
except NetworkError as e:
    st.error(f"The activity data has a problem: {e}")
    st.stop()
if risks is not None and (problems := validate_risks(risks, net)):
    st.error("Risk register problems: " + "; ".join(problems))
    st.stop()

base = baseline_comparison(net)
pdur, pcost = percentiles(res.project_duration), percentiles(res.project_cost)
cont = contingency_summary(net, res, inputs["overhead_per_month_lakh"], fin, inputs)
table = activity_table(net, res)

st.title("Construction schedule and cost risk")
st.caption("Results come from running the project thousands of times with realistic variation, risks and linked delays.")

c = st.columns(4)
c[0].metric("Plan (most-likely durations)", f"{base['most_likely_duration'] / 30:.1f} months")
c[1].metric("P50 finish", f"{pdur['P50'] / 30:.1f} months")
c[2].metric("P80 finish", f"{pdur['P80'] / 30:.1f} months", f"+{(pdur['P80'] - base['most_likely_duration']) / 30:.1f} months vs plan", delta_color="inverse")
c[3].metric("P90 finish", f"{pdur['P90'] / 30:.1f} months")
c = st.columns(4)
c[0].metric("Base cost estimate", f"₹{cont['base_estimate_lakh']:,.0f} lakh")
c[1].metric("P80 cost", f"₹{cont['P80_cost_lakh']:,.0f} lakh")
c[2].metric("Contingency needed at P80", f"{cont['contingency_P80_pct']:.1%}", f"{cont['contingency_P80_pct'] - 0.10:+.1%} vs flat 10%", delta_color="inverse")
c[3].metric("Chance a flat 10% is not enough", f"{cont['prob_cost_exceeds_flat10']:.0%}")

tabs = st.tabs(["Distributions", "Gantt", "Delay drivers", "Finance and contingency", "Mitigation", "Model ladder", "Inputs"])

def hist_cdf(x, title, unit):
    xs = np.sort(x)
    fig = go.Figure()
    fig.add_histogram(x=x, nbinsx=60, name="Runs", opacity=0.6)
    fig.add_scatter(x=xs, y=np.arange(1, len(xs) + 1) / len(xs), name="Chance of finishing within", yaxis="y2")
    for p in ("P50", "P80", "P90"):
        fig.add_vline(x=np.percentile(x, int(p[1:])), line_dash="dot", annotation_text=p)
    fig.update_layout(title=title, xaxis_title=unit, yaxis_title="Number of runs", height=400, legend=dict(orientation="h"),
                      yaxis2=dict(overlaying="y", side="right", range=[0, 1], tickformat=".0%"))
    return fig

with tabs[0]:
    st.plotly_chart(hist_cdf(res.project_duration, "Project duration", "days"), width="stretch")
    st.plotly_chart(hist_cdf(res.project_cost, "Construction cost", "₹ lakh"), width="stretch")

with tabs[1]:
    slip_p = st.radio("Show possible slip up to", [50, 80, 90], index=1, horizontal=True, format_func=lambda p: f"P{p}")
    gdf = gantt_frame(net, res, table["criticality_index"].to_numpy(), start_date, slip_p)
    st.plotly_chart(gantt_chart(gdf, slip_p), width="stretch")
    st.caption("Coloured bars are the plan using most-likely durations. Grey bars show how much later each activity "
               f"finishes in {slip_p}% of runs. Colour shows how often the activity was on the critical path.")

with tabs[2]:
    top = table.sort_values("cruciality", ascending=False).head(12).iloc[::-1]
    fig = go.Figure(go.Bar(x=top["spearman_with_project"], y=top["id"] + " " + top["activity"], orientation="h"))
    fig.update_layout(title="Activities most linked to project delay", xaxis_title="Rank correlation with project duration", height=450)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(table.sort_values("cruciality", ascending=False).round(3), hide_index=True, width="stretch")
    if risks is not None:
        st.subheader("How much each risk adds")
        if st.button("Test each risk (takes about 15 seconds)"):
            st.session_state["contrib"] = cached_contrib(acts, rho, risks, inputs["overhead_per_month_lakh"], n_iter, seed)
        if "contrib" in st.session_state:
            st.dataframe(st.session_state["contrib"].round(2), hide_index=True, width="stretch")
            st.caption("Each row: change in P80 when only that risk is switched off. A risk on an activity with float can add zero days.")

with tabs[3]:
    c = st.columns(3)
    c[0].metric("NPV of the plan", f"₹{cont['plan_npv_lakh']:,.0f} lakh")
    c[1].metric("NPV of plan + flat 10%", f"₹{cont['plan_plus_flat10_npv_lakh']:,.0f} lakh")
    c[2].metric("Simulated median NPV", f"₹{cont['sim_P50_npv_lakh']:,.0f} lakh")
    c = st.columns(3)
    c[0].metric("NPV with 80% confidence", f"₹{cont['npv_80pct_confidence_lakh']:,.0f} lakh")
    c[1].metric("Chance NPV is negative", f"{cont['prob_npv_negative']:.1%}")
    c[2].metric("Median interest during construction", f"₹{cont['sim_P50_idc_lakh']:,.0f} lakh", f"plan: ₹{cont['plan_idc_lakh']:,.0f} lakh", delta_color="off")
    c = st.columns(2)
    c[0].metric("Project IRR: plan vs median", f"{cont['plan_project_irr']:.1%} → {cont['sim_P50_project_irr']:.1%}")
    c[1].metric("Equity IRR: plan vs median", f"{cont['plan_equity_irr']:.1%} → {cont['sim_P50_equity_irr']:.1%}")
    fig = go.Figure(go.Histogram(x=fin["project_npv_lakh"], nbinsx=60))
    fig.add_vline(x=cont["plan_npv_lakh"], line_dash="dash", annotation_text="Plan")
    fig.add_vline(x=0, line_color="red")
    fig.update_layout(title="Project NPV across all runs", xaxis_title="₹ lakh", height=380)
    st.plotly_chart(fig, width="stretch")
    st.subheader("Contingency")
    st.dataframe(pd.DataFrame({
        "Budget basis": ["Base estimate", "Flat 10%", "P50", "P80", "P90"],
        "Budget (₹ lakh)": [cont["base_estimate_lakh"], cont["flat10_budget_lakh"], cont["P50_cost_lakh"], cont["P80_cost_lakh"], cont["P90_cost_lakh"]],
        "Contingency": [0, 0.10, cont["contingency_P50_pct"], cont["contingency_P80_pct"], cont["contingency_P90_pct"]],
    }).style.format({"Budget (₹ lakh)": "{:,.0f}", "Contingency": "{:.1%}"}), hide_index=True)
    with st.expander("Monthly cash flow of the plan (for checking against Excel)"):
        _, plan_table = deterministic_finance(net, inputs)
        st.dataframe(plan_table.round(2), width="stretch")

with tabs[4]:
    if risks is None:
        st.info("Switch on risk events in the sidebar to test a mitigation.")
    else:
        rid = st.selectbox("Risk to reduce", risks["id"], format_func=lambda i: f"{i}: {risks.set_index('id').loc[i, 'name']}")
        cur = float(risks.set_index("id").loc[rid, "probability"])
        c = st.columns(3)
        new_p = c[0].slider("Probability after the fix", 0.0, 1.0, round(cur / 2, 2), 0.01, help=f"Currently {cur:.0%}")
        scale = c[1].slider("Impact after the fix (share of current)", 0.0, 1.0, 1.0, 0.05)
        fix_cost = c[2].number_input("Cost of the fix (₹ lakh, paid at start)", value=20.0, step=5.0)
        out = cached_mitigation(acts, rho, risks, rid, new_p, scale, fix_cost, inputs, n_iter, seed)
        verdict = "is worth it" if out["worth_it"] else "is not worth it"
        st.subheader(f"The fix {verdict}: mean NPV changes by ₹{out['npv_gain_lakh']:,.1f} lakh")
        st.write(f"Paying up to **₹{out['max_worth_paying_lakh']:,.1f} lakh** for this fix still improves expected NPV.")
        st.dataframe(out["table"].style.format({"without fix": "{:,.2f}", "with fix": "{:,.2f}", "change": "{:+,.2f}"}), hide_index=True, width="stretch")

with tabs[5]:
    lad = cached_ladder(acts, rho if rho else {}, risks, inputs["overhead_per_month_lakh"], n_iter, seed)
    c = st.columns(2)
    c[0].plotly_chart(ladder_chart(lad, "duration_P80_days", "days"), width="stretch")
    c[1].plotly_chart(ladder_chart(lad, "cost_P80_lakh", "₹ lakh"), width="stretch")
    st.dataframe(lad.round(1), hide_index=True, width="stretch")
    st.caption("Steps 1 and 2 are single answers, so P50 and P80 are the same. Steps 3 to 5 come from the simulation. "
               "If correlation or risks are switched off in the sidebar, those steps show no change.")

with tabs[6]:
    st.subheader("Activities"); st.dataframe(acts, hide_index=True, width="stretch")
    st.subheader("Risk register"); st.dataframe(read_csv(DATA / "risk_register.csv"), hide_index=True, width="stretch")
    st.subheader("Correlation groups"); st.dataframe(read_csv(DATA / "correlation_groups.csv"), hide_index=True, width="stretch")
    st.subheader("Finance inputs used"); st.json(inputs)
