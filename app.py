# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="EDI Survey Dashboard", layout="wide")

# ------------ CONFIG: Short-name lists & Likert orders -----------------------

LIKERT_COLS_SHORT = [
    'Training/Modules', 'Comparison', 'Value', 'Communication',
    'Comfort _1','Comfort _2','Comfort _3','Comfort _4','Comfort _5',
    'Comfort _6','Comfort _7','Comfort _8','Comfort _9',
    'Q4_1','Q4_2','Q4_3','Q4_4','Q4_5','Q4_6',
    'Q2_1','Q2_2','Q2_3','Q2_4','Q2_5','Q2_6','Q2_7','Q2_8','Q2_9','Q2_10',
    'Q6_1','Q6_2','Q6_3','Q6_4','Q6_5',
    'Q8_1','Q8_2','Q8_3','Q8_4','Q8_5',
    'Q44_1','Q44_2','Q44_3','Q44_4','Q44_5','Q44_6','Q44_7',
    'Overall Issues _1','Overall Issues _2','Overall Issues _3',
    'Overall Issues _4','Overall Issues _5','Overall Issues _6',
    'Overall Issues _7','Overall Issues _8','Overall Issues _9',
    'Function','Individual Success','Promote','Transparency','Initiatives'
]

# Your Likert mapping by short column name (order = low -> high)
LIKERT_DICT = {
    'Comparison': ['Average relative to my colleagues',
                   'Much less than my colleagues',
                   'Much more than my colleagues',
                   'Slightly less than my colleagues',
                   'Slightly more than my colleagues'],
    'Value': ['Completely','Not at all','Somewhat'],
    'Communication': ['Completely','Not at all','Somewhat'],
    'Comfort _1': ['Moderately','Not at all','Slightly','Very','Very much'],
    'Comfort _2': ['Moderately','Not at all','Slightly','Very','Very much'],
    'Comfort _3': ['Moderately','Not at all','Slightly','Very','Very much'],
    'Comfort _4': ['Moderately','Not at all','Slightly','Very','Very much'],
    'Comfort _5': ['Moderately','Not at all','Slightly','Very','Very much'],
    'Comfort _6': ['Moderately','Not at all','Slightly','Very','Very much'],
    'Comfort _7': ['Moderately','Not at all','Slightly','Very','Very much'],
    'Comfort _8': ['Moderately','Not at all','Slightly','Very','Very much'],
    'Comfort _9': ['Moderately','Not at all','Slightly','Very','Very much'],
    'Q4_1': ['Neither agree nor disagree','Somewhat agree','Somewhat disagree','Strongly agree'],
    'Q4_2': ['Neither agree nor disagree','Somewhat agree','Strongly agree'],
    'Q4_3': ['Neither agree nor disagree','Somewhat agree','Strongly agree'],
    'Q4_4': ['Neither agree nor disagree','Somewhat agree','Somewhat disagree','Strongly disagree'],
    'Q4_5': ['Neither agree nor disagree','Somewhat agree','Somewhat disagree','Strongly agree','Strongly disagree'],
    'Q4_6': ['Neither agree nor disagree','Somewhat agree','Strongly agree','Strongly disagree'],
    'Q2_1': ['Neither agree nor disagree','Somewhat agree','Strongly agree','Strongly disagree'],
    'Q2_2': ['Neither agree nor disagree','Somewhat agree','Strongly agree','Strongly disagree'],
    'Q2_3': ['Neither agree nor disagree','Somewhat agree','Somewhat disagree','Strongly agree','Strongly disagree'],
    'Q2_4': ['Neither agree nor disagree','Somewhat agree','Strongly agree'],
    'Q2_5': ['Neither agree nor disagree','Somewhat agree','Somewhat disagree','Strongly agree'],
    'Q2_6': ['Neither agree nor disagree','Somewhat agree','Strongly agree'],
    'Q2_7': ['Neither agree nor disagree','Somewhat agree','Somewhat disagree','Strongly disagree'],
    'Q2_8': ['Neither agree nor disagree','Somewhat agree','Strongly agree'],
    'Q2_9': ['Neither agree nor disagree','Somewhat agree','Strongly agree'],
    'Q2_10': ['Neither agree nor disagree','Somewhat agree','Strongly agree','Strongly disagree'],
    'Q6_1': ['Neither agree nor disagree','Somewhat agree','Strongly agree','Strongly disagree'],
    'Q6_2': ['Neither agree nor disagree','Somewhat agree','Strongly agree','Strongly disagree'],
    'Q6_3': ['Neither agree nor disagree','Somewhat agree','Strongly agree','Strongly disagree'],
    'Q6_4': ['Neither agree nor disagree','Somewhat agree','Somewhat disagree','Strongly agree','Strongly disagree'],
    'Q6_5': ['Neither agree nor disagree','Somewhat disagree','Strongly disagree'],
    'Q8_1': ['Neither agree nor disagree','Somewhat agree','Strongly agree'],
    'Q8_2': ['Neither agree nor disagree','Somewhat agree','Strongly agree'],
    'Q8_3': ['Neither agree nor disagree','Somewhat agree','Somewhat disagree','Strongly agree'],
    'Q8_4': ['Neither agree nor disagree','Somewhat agree','Somewhat disagree','Strongly agree','Strongly disagree'],
    'Q8_5': ['Neither agree nor disagree','Somewhat agree','Somewhat disagree','Strongly agree','Strongly disagree'],
    'Q44_1': ['Neither agree nor disagree','Somewhat agree','Strongly Agree'],
    'Q44_2': ['Neither agree nor disagree','Somewhat agree','Strongly Agree'],
    'Q44_3': ['Neither agree nor disagree','Somewhat agree','Strongly Agree'],
    'Q44_4': ['Neither agree nor disagree','Somewhat agree','Strongly Agree'],
    'Q44_5': ['Neither agree nor disagree','Somewhat agree','Strongly Agree'],
    'Q44_6': ['Neither agree nor disagree','Somewhat disagree','Strongly Agree','Strongly Disagree'],
    'Q44_7': ['Neither agree nor disagree','Somewhat agree','Strongly Agree'],
    'Overall Issues _1': ['Moderately','Not at all','Often'],
    'Overall Issues _2': ['Moderately','Not at all','Often','Slightly'],
    'Overall Issues _3': ['Moderately','Not at all','Slightly'],
    'Overall Issues _4': ['Moderately','Not at all','Slightly'],
    'Overall Issues _5': ['Moderately','Not at all','Slightly'],
    'Overall Issues _6': ['Moderately','Not at all','Often','Slightly'],
    'Overall Issues _7': ['Moderately','Not at all'],
    'Overall Issues _8': ['Moderately','Not at all','Slightly'],
    'Overall Issues _9': ['Not at all','Slightly'],
    'Function': ['Completely','Moderately','Not at all','Slightly','Very'],
    'Individual Success': ['Moderately','Not at all','Often','Slightly','Very Often'],
    'Promote': ['Completely','Moderately','Not at all','Very'],
    'Transparency': ['Completely','Moderately','Not at all','Very'],
    'Initiatives': ['About right','Far too little','Slightly too little','Slightly too much'],
    'Training/Modules': ['1-2 per semester','3-4 per semester','5+ per semester']  # adjust if needed
}

CONTROL_SHORT = [
    'Age','Department','Ethnicity','Sexual Orientation','Disability Status',
    'Employment Status','Trainee','Student/faculty','Domestic/internat'
]

# ------------ Helpers --------------------------------------------------------

def short_name(col:str) -> str:
    """Return the part before ' | ' (or the whole string if no delimiter)."""
    if not isinstance(col, str):
        col = str(col)
    parts = col.split(' | ', 1)
    return parts[0].strip()

def normalize_category(value):
    if pd.isna(value):
        return np.nan
    return str(value).strip().lower()

def coerce_to_ordered(series: pd.Series, ordered_levels):
    """Case-insensitive mapping to canonical ordered categories."""
    lut = {normalize_category(x): x for x in ordered_levels}
    mapped = series.map(lambda v: lut.get(normalize_category(v), np.nan))
    return pd.Categorical(mapped, categories=ordered_levels, ordered=True)

def encode_numeric_from_cat(series: pd.Series):
    if not pd.api.types.is_categorical_dtype(series):
        return series
    return series.cat.codes.replace(-1, np.nan) + 1

def percent_stack_df(series: pd.Series, levels):
    s = series.dropna()
    n = len(s)
    if n == 0:
        return pd.DataFrame(columns=['category','count','percent'])
    counts = s.value_counts().reindex(levels, fill_value=0)
    perc = (counts / n * 100).round(1)
    return pd.DataFrame({'category': counts.index, 'count': counts.values, 'percent': perc.values})

# ------------ UI: Load data --------------------------------------------------

st.title("EDI Survey Dashboard")

st.sidebar.header("1) Load CSV")
file = st.sidebar.file_uploader("Upload the exported CSV", type=["csv"])
if file is None:
    st.info("Upload the CSV with headers like `Short | Full question text`.")
    st.stop()

df_raw = pd.read_csv(file)

# Map short -> full column name present in this file
short_to_full = {short_name(c): c for c in df_raw.columns}

# Helper to fetch a DF column by short name (returns None if absent)
def col_by_short(short):
    return short_to_full.get(short)

# ------------ Prepare Likert & control columns -------------------------------

# Build list of available Likert columns using short names
available_likert_pairs = [(s, col_by_short(s)) for s in LIKERT_COLS_SHORT if col_by_short(s) in df_raw.columns]
if not available_likert_pairs:
    st.warning("No Likert columns found based on short names.")
available_likert_full = [full for s, full in available_likert_pairs]

# Coerce ordered categories for Likert cols
df = df_raw.copy()
for s, full in available_likert_pairs:
    levels = LIKERT_DICT.get(s)
    if levels:
        df[full] = coerce_to_ordered(df[full], levels)

# Numeric version (for boxplots)
numeric_df = df.copy()
for s, full in available_likert_pairs:
    if pd.api.types.is_categorical_dtype(df[full]):
        numeric_df[full] = encode_numeric_from_cat(df[full])

# Controls: detect full names
control_map = {s: col_by_short(s) for s in CONTROL_SHORT if col_by_short(s) in df.columns}

# ------------ Filters --------------------------------------------------------

st.sidebar.header("2) Filters")
filtered = df.copy()

# Age as range slider if present
age_full = control_map.get('Age')
if age_full:
    # coerce to numeric (ignore non-numeric)
    age_num = pd.to_numeric(df[age_full], errors='coerce')
    df['_age_num'] = age_num
    min_age = int(np.nanmin(age_num)) if age_num.notna().any() else 0
    max_age = int(np.nanmax(age_num)) if age_num.notna().any() else 100
    age_lo, age_hi = st.sidebar.slider("Age range", min_value=min_age, max_value=max_age, value=(min_age, max_age))
    filtered = filtered[(df['_age_num'] >= age_lo) & (df['_age_num'] <= age_hi)]

# Multi-select for the rest
for s, full in control_map.items():
    if s == 'Age':
        continue
    vals = sorted([v for v in df[full].dropna().unique()], key=lambda x: str(x))
    if len(vals) == 0:
        continue
    sel = st.sidebar.multiselect(s, options=vals, default=vals)
    filtered = filtered[filtered[full].isin(sel)]

# ------------ Faceting & chart options --------------------------------------

st.sidebar.header("3) Chart options")
chart_type = st.sidebar.radio(
    "Display",
    ["100% Stacked Horizontal Bars (all items)", "Boxplots (all items)"],
    index=0
)
show_counts = st.sidebar.checkbox("Show counts in hover", value=True)

st.sidebar.header("4) Optional faceting")
facet_options = ["(none)"] + [k for k in CONTROL_SHORT if k in control_map]
facet_by_short = st.sidebar.selectbox("Facet/group by (optional)", options=facet_options, index=0)
facet_full = control_map.get(facet_by_short) if facet_by_short != "(none)" else None

st.markdown(f"**Filtered sample size:** {len(filtered):,} (of {len(df):,})")

# ------------ Plot functions (faceted segments & side-by-side) ---------------

def plot_100_stacked_for_item(df_in, short_name_key, full_col):
    """Plot 100% stacked bar for a given item, using full_col for title, short_name for axis."""
    levels = LIKERT_DICT.get(short_name_key, None)
    if levels is None:
        st.info(f"Skipping {short_name_key}: no Likert order defined.")
        return

    if facet_full:
        # Faceted segmented bars (one per group)
        rows = []
        for gname, gdf in df_in.groupby(facet_full, dropna=False):
            d = percent_stack_df(gdf[full_col], levels)
            if not d.empty:
                d['facet'] = str(gname)
                rows.append(d)
        if not rows:
            st.info(f"No data for {short_name_key} after filters.")
            return
        data = pd.concat(rows, ignore_index=True)
        facet_order = list(data["facet"].dropna().unique())

        fig = px.bar(
            data,
            x="percent",
            y="facet",
            color="category",
            orientation="h",
            category_orders={"category": levels, "facet": facet_order},
            hover_data=(["count","percent"] if show_counts else ["percent"]),
            labels={
                "percent": "%",
                "facet": facet_by_short,
                "category": "Response"
            },
        )
        n_total = df_in[full_col].notna().sum()
        fig.update_layout(
            title=f"{full_col} — segmented by {facet_by_short} (n={n_total})",
            barmode="stack",
            xaxis=dict(ticksuffix="%", range=[0, 100], automargin=True),
            yaxis=dict(title=short_name_key, autorange="reversed", automargin=True),
            legend_title="",
            margin=dict(l=10, r=10, t=60, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        # Single 100% stacked bar (no facet)
        data = percent_stack_df(df_in[full_col], levels)
        if data.empty:
            st.info(f"No data for {short_name_key} after filters.")
            return
        fig = px.bar(
            data,
            x="percent",
            y=[short_name_key]*len(data),
            color="category",
            orientation="h",
            category_orders={"category": levels},
            hover_data=(["count","percent"] if show_counts else ["percent"]),
            labels={"percent": "%", "category": "Response"},
        )
        fig.update_layout(
            title=f"{full_col} (n={df_in[full_col].notna().sum()})",
            barmode="stack",
            xaxis=dict(ticksuffix="%", range=[0, 100]),
            yaxis=dict(title=short_name_key, showticklabels=False),
            legend_title="",
            margin=dict(l=10, r=10, t=60, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)



def plot_box_for_item(df_in, short_name_key, full_col):
    """Plot boxplot for a given item, full_col for title, short_name for axis."""
    if full_col not in numeric_df.columns:
        st.info(f"Skipping {short_name_key}: numeric encoding missing.")
        return

    dsub = numeric_df.loc[df_in.index, [full_col]].rename(columns={full_col: "score"}).dropna()
    if dsub.empty:
        st.info(f"No data for {short_name_key} after filters.")
        return

    if facet_full:
        dsub[facet_by_short] = df.loc[dsub.index, facet_full].astype(str)
        facet_order = list(dsub[facet_by_short].dropna().unique())
        fig = px.box(
            dsub,
            x=facet_by_short,
            y="score",
            points="all",
            category_orders={facet_by_short: facet_order},
            labels={
                "score": f"{short_name_key} (1 = first level in order)",
                "facet_by_short": facet_by_short
            },
        )
        fig.update_layout(
            title=f"{full_col} — Side-by-side boxplots by {facet_by_short} (n={len(dsub)})",
            xaxis=dict(title=facet_by_short, tickangle=-45, automargin=True),
            yaxis=dict(title=short_name_key, automargin=True),
            margin=dict(l=10, r=10, t=60, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        fig = px.box(
            dsub,
            x="score",
            points="all",
            labels={"score": f"{short_name_key} (1 = first level in order)"},
        )
        fig.update_layout(
            title=f"{full_col} — Boxplot (n={len(dsub)})",
            yaxis=dict(title=short_name_key, showticklabels=False),
            margin=dict(l=10, r=10, t=60, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)


# ------------ Render ----------------------------------------------------------

st.caption("Note: headers are parsed as `Short name | Full question text`; filters & Likert mappings use **short names**.")

if chart_type.startswith("100%"):
    st.subheader("100% Stacked Horizontal Bar Charts (segmented if faceted)")
    for s, full in available_likert_pairs:
        plot_100_stacked_for_item(filtered, s, full)
else:
    st.subheader("Side-by-side Boxplots (rotated x-axis labels when faceted)")
    for s, full in available_likert_pairs:
        plot_box_for_item(filtered, s, full)

# # app.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px

# st.set_page_config(page_title="Survey Likert Dashboard", layout="wide")

# # --- CONFIG ------------------------------------------------------------------

# # All Likert columns you listed
# LIKERT_COLS = [
#     'Training/Modules', 'Comparison', 'Value', 'Communication',
#     'Comfort _1', 'Comfort _2', 'Comfort _3', 'Comfort _4', 'Comfort _5',
#     'Comfort _6', 'Comfort _7', 'Comfort _8', 'Comfort _9',
#     'Q4_1', 'Q4_2', 'Q4_3', 'Q4_4', 'Q4_5', 'Q4_6',
#     'Q2_1', 'Q2_2', 'Q2_3', 'Q2_4', 'Q2_5', 'Q2_6', 'Q2_7', 'Q2_8', 'Q2_9', 'Q2_10',
#     'Q6_1', 'Q6_2', 'Q6_3', 'Q6_4', 'Q6_5',
#     'Q8_1', 'Q8_2', 'Q8_3', 'Q8_4', 'Q8_5',
#     'Q44_1', 'Q44_2', 'Q44_3', 'Q44_4', 'Q44_5', 'Q44_6', 'Q44_7',
#     'Overall Issues _1', 'Overall Issues _2', 'Overall Issues _3',
#     'Overall Issues _4', 'Overall Issues _5', 'Overall Issues _6',
#     'Overall Issues _7', 'Overall Issues _8', 'Overall Issues _9'
# ]

# # Your provided Likert levels (order matters; 1 = first, n = last)
# LIKERT_DICT = {
#     'Comparison': ['Average relative to my colleagues',
#                    'Much less than my colleagues',
#                    'Much more than my colleagues',
#                    'Slightly less than my colleagues',
#                    'Slightly more than my colleagues'],
#     'Value': ['Completely', 'Not at all', 'Somewhat'],
#     'Communication': ['Completely', 'Not at all', 'Somewhat'],
#     'Comfort _1': ['Moderately', 'Not at all', 'Slightly', 'Very', 'Very much'],
#     'Comfort _2': ['Moderately', 'Not at all', 'Slightly', 'Very', 'Very much'],
#     'Comfort _3': ['Moderately', 'Not at all', 'Slightly', 'Very', 'Very much'],
#     'Comfort _4': ['Moderately', 'Not at all', 'Slightly', 'Very', 'Very much'],
#     'Comfort _5': ['Moderately', 'Not at all', 'Slightly', 'Very', 'Very much'],
#     'Comfort _6': ['Moderately', 'Not at all', 'Slightly', 'Very', 'Very much'],
#     'Comfort _7': ['Moderately', 'Not at all', 'Slightly', 'Very', 'Very much'],
#     'Comfort _8': ['Moderately', 'Not at all', 'Slightly', 'Very', 'Very much'],
#     'Comfort _9': ['Moderately', 'Not at all', 'Slightly', 'Very', 'Very much'],
#     'Q4_1': ['Neither agree nor disagree', 'Somewhat agree', 'Somewhat disagree', 'Strongly agree'],
#     'Q4_2': ['Neither agree nor disagree', 'Somewhat agree', 'Strongly agree'],
#     'Q4_3': ['Neither agree nor disagree', 'Somewhat agree', 'Strongly agree'],
#     'Q4_4': ['Neither agree nor disagree', 'Somewhat agree', 'Somewhat disagree', 'Strongly disagree'],
#     'Q4_5': ['Neither agree nor disagree', 'Somewhat agree', 'Somewhat disagree', 'Strongly agree', 'Strongly disagree'],
#     'Q4_6': ['Neither agree nor disagree', 'Somewhat agree', 'Strongly agree', 'Strongly disagree'],
#     'Q2_1': ['Neither agree nor disagree', 'Somewhat agree', 'Strongly agree', 'Strongly disagree'],
#     'Q2_2': ['Neither agree nor disagree', 'Somewhat agree', 'Strongly agree', 'Strongly disagree'],
#     'Q2_3': ['Neither agree nor disagree', 'Somewhat agree', 'Somewhat disagree', 'Strongly agree', 'Strongly disagree'],
#     'Q2_4': ['Neither agree nor disagree', 'Somewhat agree', 'Strongly agree'],
#     'Q2_5': ['Neither agree nor disagree', 'Somewhat agree', 'Somewhat disagree', 'Strongly agree'],
#     'Q2_6': ['Neither agree nor disagree', 'Somewhat agree', 'Strongly agree'],
#     'Q2_7': ['Neither agree nor disagree', 'Somewhat agree', 'Somewhat disagree', 'Strongly disagree'],
#     'Q2_8': ['Neither agree nor disagree', 'Somewhat agree', 'Strongly agree'],
#     'Q2_9': ['Neither agree nor disagree', 'Somewhat agree', 'Strongly agree'],
#     'Q2_10': ['Neither agree nor disagree', 'Somewhat agree', 'Strongly agree', 'Strongly disagree'],
#     'Q6_1': ['Neither agree nor disagree', 'Somewhat agree', 'Strongly agree', 'Strongly disagree'],
#     'Q6_2': ['Neither agree nor disagree', 'Somewhat agree', 'Strongly agree', 'Strongly disagree'],
#     'Q6_3': ['Neither agree nor disagree', 'Somewhat agree', 'Strongly agree', 'Strongly disagree'],
#     'Q6_4': ['Neither agree nor disagree', 'Somewhat agree', 'Somewhat disagree', 'Strongly agree', 'Strongly disagree'],
#     'Q6_5': ['Neither agree nor disagree', 'Somewhat disagree', 'Strongly disagree'],
#     'Q8_1': ['Neither agree nor disagree', 'Somewhat agree', 'Strongly agree'],
#     'Q8_2': ['Neither agree nor disagree', 'Somewhat agree', 'Strongly agree'],
#     'Q8_3': ['Neither agree nor disagree', 'Somewhat agree', 'Somewhat disagree', 'Strongly agree'],
#     'Q8_4': ['Neither agree nor disagree', 'Somewhat agree', 'Somewhat disagree', 'Strongly agree', 'Strongly disagree'],
#     'Q8_5': ['Neither agree nor disagree', 'Somewhat agree', 'Somewhat disagree', 'Strongly agree', 'Strongly disagree'],
#     'Q44_1': ['Neither agree nor disagree', 'Somewhat agree', 'Strongly Agree'],
#     'Q44_2': ['Neither agree nor disagree', 'Somewhat agree', 'Strongly Agree'],
#     'Q44_3': ['Neither agree nor disagree', 'Somewhat agree', 'Strongly Agree'],
#     'Q44_4': ['Neither agree nor disagree', 'Somewhat agree', 'Strongly Agree'],
#     'Q44_5': ['Neither agree nor disagree', 'Somewhat agree', 'Strongly Agree'],
#     'Q44_6': ['Neither agree nor disagree', 'Somewhat disagree', 'Strongly Agree', 'Strongly Disagree'],
#     'Q44_7': ['Neither agree nor disagree', 'Somewhat agree', 'Strongly Agree'],
#     'Overall Issues _1': ['Moderately', 'Not at all', 'Often'],
#     'Overall Issues _2': ['Moderately', 'Not at all', 'Often', 'Slightly'],
#     'Overall Issues _3': ['Moderately', 'Not at all', 'Slightly'],
#     'Overall Issues _4': ['Moderately', 'Not at all', 'Slightly'],
#     'Overall Issues _5': ['Moderately', 'Not at all', 'Slightly'],
#     'Overall Issues _6': ['Moderately', 'Not at all', 'Often', 'Slightly'],
#     'Overall Issues _7': ['Moderately', 'Not at all'],
#     'Overall Issues _8': ['Moderately', 'Not at all', 'Slightly'],
#     'Overall Issues _9': ['Not at all', 'Slightly'],
#     'Function': ['Completely', 'Moderately', 'Not at all', 'Slightly', 'Very'],
#     'Individual Success': ['Moderately', 'Not at all', 'Often', 'Slightly', 'Very Often'],
#     'Promote': ['Completely', 'Moderately', 'Not at all', 'Very'],
#     'Transparency': ['Completely', 'Moderately', 'Not at all', 'Very'],
#     'Initiatives': ['About right', 'Far too little', 'Slightly too little', 'Slightly too much'],
#     'Training/Modules': ['1-2 per semester', '3-4 per semester', '5+ per semester']  # OPTIONAL: adjust if you have defined categories
# }

# # Filter control columns (multiple choice)
# CONTROL_COLS = [
#     'Age', 'Department', 'Ethnicity', 'Sexual Orientation', 'Disability Status',
#     'Employment Status', 'Trainee', 'Student/faculty', 'Domestic/internat'
# ]

# # --- HELPERS -----------------------------------------------------------------

# def normalize_category(value):
#     """Return a safe, normalized text for matching (handles case/whitespace)."""
#     if pd.isna(value):
#         return np.nan
#     return str(value).strip().lower()

# def coerce_to_ordered_categories(series: pd.Series, ordered_levels):
#     """Map raw text to exact category labels using case-insensitive matching."""
#     # Build lookup dict from normalized category -> canonical label
#     lut = {normalize_category(x): x for x in ordered_levels}
#     # Map series values to canonical labels if possible
#     mapped = series.map(lambda v: lut.get(normalize_category(v), np.nan))
#     cat = pd.Categorical(mapped, categories=ordered_levels, ordered=True)
#     return pd.Series(cat, index=series.index)

# def encode_likert_numeric(series: pd.Series):
#     """Convert ordered categorical to 1..n numeric (NaNs preserved)."""
#     if not pd.api.types.is_categorical_dtype(series):
#         return series
#     codes = series.cat.codes.replace(-1, np.nan) + 1
#     return codes

# def percent_stack_df(series: pd.Series, levels):
#     """Return a tidy DF with counts and percents per category for one item."""
#     s = series.dropna()
#     total = len(s)
#     if total == 0:
#         return pd.DataFrame(columns=['category','count','percent'])
#     counts = s.value_counts().reindex(levels, fill_value=0)
#     perc = (counts / total * 100).round(1)
#     out = pd.DataFrame({'category': counts.index, 'count': counts.values, 'percent': perc.values})
#     return out

# # --- SIDEBAR / DATA INPUT ----------------------------------------------------

# st.title("Survey Likert Dashboard")

# st.sidebar.header("1) Load data")
# uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
# if uploaded is not None:
#     df = pd.read_csv(uploaded)
# else:
#     # Fallback path; replace with your file or require upload
#     try:
#         df = pd.read_csv("PATH_TO_YOUR_DATA.csv")
#         st.sidebar.info("Loaded local CSV: PATH_TO_YOUR_DATA.csv")
#     except Exception:
#         st.sidebar.warning("Please upload a CSV file to proceed.")
#         st.stop()

# # --- PREP: Ensure columns exist and coerce categories ------------------------

# missing_likerts = [c for c in LIKERT_COLS if c not in df.columns]
# if missing_likerts:
#     st.warning(f"Missing Likert columns (skipped): {', '.join(missing_likerts)}")

# available_likerts = [c for c in LIKERT_COLS if c in df.columns]

# # Coerce each Likert to ordered categorical based on LIKERT_DICT (if available)
# for col in available_likerts:
#     levels = LIKERT_DICT.get(col, None)
#     if levels:
#         df[col] = coerce_to_ordered_categories(df[col], levels)

# # Numeric version for boxplots
# numeric_df = df.copy()
# for col in available_likerts:
#     if pd.api.types.is_categorical_dtype(df[col]):
#         numeric_df[col] = encode_likert_numeric(df[col])

# # --- SIDEBAR: FILTERS --------------------------------------------------------

# st.sidebar.header("2) Filters (multiple choice)")
# filtered = df.copy()
# for ctrl in CONTROL_COLS:
#     if ctrl in df.columns:
#         vals = df[ctrl].dropna().unique().tolist()
#         vals = sorted(vals, key=lambda x: str(x))
#         selection = st.sidebar.multiselect(ctrl, options=vals, default=vals)
#         filtered = filtered[filtered[ctrl].isin(selection)]
#     else:
#         st.sidebar.caption(f"⚠️ Column not found: {ctrl}")

# st.sidebar.header("3) Chart options")
# chart_type = st.sidebar.radio(
#     "Display",
#     ["100% Stacked Horizontal Bars (all items)", "Boxplots (all items)"],
#     index=0
# )
# show_counts = st.sidebar.checkbox("Show raw counts in hover", value=True)

# st.sidebar.header("4) Optional faceting")
# facet_by = st.sidebar.selectbox(
#     "Facet/group by (optional)",
#     options=["(none)"] + [c for c in CONTROL_COLS if c in df.columns],
#     index=0
# )

# st.markdown(
#     f"**Filtered sample size:** {len(filtered):,} respondents "
#     f"(of {len(df):,} total)"
# )

# # --- CHARTING ----------------------------------------------------------------

# def plot_100_stacked_for_item(df_in, col):
#     levels = LIKERT_DICT.get(col, None)
#     if levels is None:
#         st.info(f"Skipping {col}: no level order defined.")
#         return

#     # Build tidy table for all facet levels at once
#     if facet_by != "(none)" and facet_by in df_in.columns:
#         grp = df_in.groupby(facet_by, dropna=False)
#         rows = []
#         for gname, gdf in grp:
#             d = percent_stack_df(gdf[col], levels)
#             if not d.empty:
#                 d["facet"] = str(gname)
#                 rows.append(d)
#         if not rows:
#             st.info(f"No data for {col} after filters.")
#             return
#         data = pd.concat(rows, ignore_index=True)

#         # Ensure facet order is stable (by appearance)
#         facet_order = list(data["facet"].dropna().unique())

#         fig = px.bar(
#             data,
#             x="percent",
#             y="facet",
#             color="category",
#             orientation="h",
#             category_orders={"category": levels, "facet": facet_order},
#             hover_data=(["count", "percent"] if show_counts else ["percent"]),
#             labels={"percent": "%", "facet": facet_by, "category": "Response"},
#         )
#         n_total = df_in[col].notna().sum()
#         fig.update_layout(
#             title=f"{col} — segmented by {facet_by} (n={n_total})",
#             barmode="stack",
#             xaxis=dict(ticksuffix="%", range=[0, 100]),
#             yaxis=dict(autorange="reversed", tickangle=0, automargin=True),  # keep bars descending; allow long labels
#             legend_title="",
#             margin=dict(l=10, r=10, t=60, b=10),
#         )
#         st.plotly_chart(fig, use_container_width=True)

#     else:
#         # No facet: single 100% bar
#         data = percent_stack_df(df_in[col], levels)
#         if data.empty:
#             st.info(f"No data for {col} after filters.")
#             return
#         fig = px.bar(
#             data,
#             x="percent",
#             y=[col]*len(data),
#             color="category",
#             orientation="h",
#             category_orders={"category": levels},
#             hover_data=(["count", "percent"] if show_counts else ["percent"]),
#             labels={"percent": "%", "category": "Response"},
#         )
#         fig.update_layout(
#             title=f"{col} (n={df_in[col].notna().sum()})",
#             barmode="stack",
#             xaxis=dict(ticksuffix="%", range=[0, 100]),
#             yaxis=dict(showticklabels=False),
#             legend_title="",
#             margin=dict(l=10, r=10, t=60, b=10),
#         )
#         st.plotly_chart(fig, use_container_width=True)


# def plot_box_for_item(df_in, col):
#     # Use numeric encoded values for boxplot
#     if col not in numeric_df.columns:
#         st.info(f"Skipping {col}: missing in numeric DF.")
#         return

#     # Subset to filtered indices, keep score
#     dsub = numeric_df.loc[df_in.index, [col]].copy()
#     dsub = dsub.rename(columns={col: "score"})
#     dsub = dsub.dropna(subset=["score"])
#     if dsub.empty:
#         st.info(f"No data for {col} after filters.")
#         return

#     # With a facet: side-by-side boxes across facet levels
#     if facet_by != "(none)" and facet_by in df.columns:
#         dsub[facet_by] = df.loc[dsub.index, facet_by].astype(str)
#         # Order facets by appearance
#         facet_order = list(dsub[facet_by].dropna().unique())

#         fig = px.box(
#             dsub,
#             x=facet_by,     # categories along x
#             y="score",      # numeric scores on y
#             points="all",   # show jittered points
#             category_orders={facet_by: facet_order},
#             labels={"score": f"{col} (1 = first level in order)"},
#         )
#         fig.update_layout(
#             title=f"{col} — Side-by-side boxplots by {facet_by} (n={len(dsub)})",
#             xaxis=dict(tickangle=-45, automargin=True),   # rotate to avoid overlap
#             yaxis=dict(automargin=True),
#             margin=dict(l=10, r=10, t=60, b=10),
#         )
#         st.plotly_chart(fig, use_container_width=True)

#     else:
#         # No facet: single box (horizontal distribution optional)
#         fig = px.box(
#             dsub,
#             x="score",
#             points="all",
#             labels={"score": f"{col} (1 = first level in order)"},
#         )
#         fig.update_layout(
#             title=f"{col} — Boxplot (n={len(dsub)})",
#             yaxis=dict(showticklabels=False),
#             margin=dict(l=10, r=10, t=60, b=10),
#         )
#         st.plotly_chart(fig, use_container_width=True)

# # --- RENDER ------------------------------------------------------------------

# with st.expander("About the scoring & orders", expanded=False):
#     st.write("""
# - **Orders** are taken from your `LIKERT_DICT`. The **first** label in each list is treated as **1**, the last as the **highest**.
# - 100% stacked bars use these label orders left-to-right in the legend.
# - Boxplots reflect numeric encoding (1..n). If a response string differs by case/whitespace,
#   it is auto-matched case-insensitively to the closest canonical label.
# - Missing or unmatched responses are treated as NA (excluded from charts).
#     """)

# if chart_type.startswith("100%"):
#     st.subheader("100% Stacked Horizontal Bar Charts (all Likert items)")
#     for col in available_likerts:
#         plot_100_stacked_for_item(filtered, col)
# else:
#     st.subheader("Boxplots (all Likert items)")
#     for col in available_likerts:
#         plot_box_for_item(filtered, col)

# st.caption("Tip: Use the sidebar to filter respondents and to facet by a demographic/control.")
