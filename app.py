import io
import re
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer


# =========================================================
# FeedbackLens AI — Feedback Intelligence Platform
# =========================================================

st.set_page_config(
    page_title="FeedbackLens AI | Feedback Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Design ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#253247; --muted:#7b8798; --line:#e8edf2; --green:#b4e63d; --green-dark:#7caa16; --navy:#26364a; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:#f5f7fa; color:var(--ink); }
[data-testid="stHeader"] { background:rgba(245,247,250,.94); }
[data-testid="stSidebar"] { background:#fff; border-right:1px solid #e7ebf0; min-width:250px; }
/* Keep the sidebar compact instead of spreading its items down the page. */
[data-testid="stSidebar"] > div:first-child { padding-top:1rem; }
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding:1rem 1rem 1.25rem; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap:.65rem; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { margin-bottom:.25rem; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { padding:.15rem 0; }
.block-container { max-width:1600px; padding:1.5rem 2rem 2.5rem; }
h1,h2,h3 { font-family:'Manrope',sans-serif !important; color:#26364a !important; letter-spacing:-.4px; }
p, label, span { color:#5d6b7d; }
[data-testid="stCaptionContainer"] p { color:#697789 !important; }\n.js-plotly-plot .gtitle, .js-plotly-plot .xtitle, .js-plotly-plot .ytitle { fill:#344256 !important; }\n.js-plotly-plot .xtick text, .js-plotly-plot .ytick text, .js-plotly-plot .legend text, .js-plotly-plot text { fill:#526174 !important; color:#526174 !important; }
[data-testid="stAlert"] p { color:#344256 !important; }
.stMultiSelect, .stSelectbox { color:#344256 !important; }
.hero { background:transparent; border:0; padding:0; margin:0 0 18px; }
.eyebrow { color:#6f9715; font-size:11px; letter-spacing:.5px; font-weight:700; text-transform:none; }
.hero h1 { font-size:30px; margin:4px 0; color:#26364a !important; }
.hero p { font-size:13px; line-height:1.5; color:#7b8798; margin:0; }
.metric-card { background:#fff; border:1px solid #e9edf2; border-radius:13px; padding:17px 18px; min-height:112px; box-shadow:0 2px 8px rgba(28,44,64,.025); }
.metric-label { color:#7d8998; font-size:11px; text-transform:none; letter-spacing:0; font-weight:700; }
.metric-value { color:#26364a; font-family:'Manrope',sans-serif; font-size:27px; font-weight:800; margin-top:8px; }
.metric-note { font-size:11px; color:#8a96a5; margin-top:3px; }
.insight-card { background:#fff; border:1px solid #e8edf2; border-radius:12px; padding:15px 16px; margin-bottom:10px; box-shadow:0 2px 8px rgba(28,44,64,.025); }
.insight-title { font-size:14px; font-weight:800; color:#29384c; margin-bottom:6px; }
.insight-body { font-size:12px; color:#6f7d8d; line-height:1.65; }
.small-muted { color:#8a96a5; font-size:11px; }
div.stButton > button { border-radius:8px; min-height:40px; font-weight:700; border:1px solid #e3e8ed; }
div.stButton > button[kind="primary"] { background:#b4e63d; border-color:#b4e63d; color:#34451c; }
div[data-testid="stFileUploader"] { background:#fff; border:1px dashed #cbd5df; border-radius:12px; padding:12px; }
[data-testid="stMetric"] { background:#fff; border:1px solid #e8edf2; border-radius:12px; padding:12px; }
hr { border-color:#e8edf2; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius:12px; }
.stSelectbox div[data-baseweb="select"] > div, .stMultiSelect div[data-baseweb="select"] > div { background:#fff; border-color:#e0e6ec; border-radius:8px; }
[data-testid="stDataFrame"] { border:1px solid #e8edf2; border-radius:10px; overflow:hidden; }

/* Readable light table styling for keyword counts and report summaries. */
[data-testid="stDataFrame"] {
    border: 1px solid #e3e9ef !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    background: #ffffff !important;
}

/* Editable reply draft: white background, dark text, and a subtle green focus state. */
[data-testid="stTextArea"] textarea {
    background: #ffffff !important;
    color: #26364a !important;
    -webkit-text-fill-color: #26364a !important;
    border: 1px solid #d7e0e8 !important;
    border-radius: 10px !important;
    box-shadow: none !important;
    caret-color: #26364a !important;
    line-height: 1.6 !important;
}
[data-testid="stTextArea"] textarea::placeholder {
    color: #8995a4 !important;
    -webkit-text-fill-color: #8995a4 !important;
}
[data-testid="stTextArea"] textarea:focus {
    border: 1px solid #91bd27 !important;
    box-shadow: 0 0 0 2px rgba(180,230,61,.20) !important;
    outline: none !important;
}
[data-testid="stTextArea"] label,
[data-testid="stTextArea"] label p {
    color: #526174 !important;
    font-weight: 600 !important;
}
@media (max-width:768px) { .block-container{padding:1rem;} .hero h1{font-size:25px;} }
</style>
""", unsafe_allow_html=True)

analyzer = SentimentIntensityAnalyzer()

THEME_RULES = {
    "Customer Service": [
        "support", "staff", "service", "agent", "rude", "helpful",
        "response", "representative", "refund", "unhelpful", "customer care",
    ],
    "Delivery & Logistics": [
        "delivery", "delivered", "shipping", "shipment", "late", "delay",
        "delayed", "package", "courier", "arrived", "tracking",
    ],
    "Product Quality": [
        "quality", "broken", "damaged", "defective", "durable", "material",
        "faulty", "poor quality", "excellent quality", "performance",
    ],
    "Pricing & Value": [
        "price", "pricing", "expensive", "cheap", "cost", "affordable",
        "value", "overpriced", "money",
    ],
    "Usability & Experience": [
        "website", "application", "app", "interface", "easy to use",
        "difficult to use", "confusing", "navigation", "slow", "crash", "loading",
    ],
    "Teaching & Learning": [
        "teacher", "teaching", "faculty", "lecture", "classroom", "curriculum",
        "course", "learning", "assignment", "lab",
    ],
    "Facilities & Cleanliness": [
        "clean", "cleanliness", "facility", "facilities", "toilet", "washroom",
        "classroom", "library", "canteen", "infrastructure",
    ],
}

THEME_RECOMMENDATIONS = {
    "Customer Service": "Review complaint-handling procedures, response times, and staff training.",
    "Delivery & Logistics": "Investigate delayed deliveries, tracking accuracy, and delivery-partner performance.",
    "Product Quality": "Inspect recurring defects, review quality checks, and investigate reported failures.",
    "Pricing & Value": "Review price-to-value concerns, communicate included features, and compare alternatives.",
    "Usability & Experience": "Test the user journey, investigate slow pages or crashes, and simplify confusing steps.",
    "Teaching & Learning": "Review the affected course, collect specific student examples, and evaluate teaching support.",
    "Facilities & Cleanliness": "Inspect the reported facilities and establish a documented maintenance or cleanliness checklist.",
    "Other / Unclassified": "Manually review frequent negative comments to discover issues not covered by current theme rules.",
}

SAMPLE_REVIEWS = [
    ("The staff were helpful and the service was excellent.", 5),
    ("The delivery arrived two days late.", 2),
    ("The product quality is excellent.", 5),
    ("Customer support did not answer my complaint.", 1),
    ("The price is expensive for what you receive.", 2),
    ("The application is easy to use.", 5),
    ("The website is slow and confusing.", 2),
    ("The package arrived damaged.", 1),
    ("The teacher explains concepts clearly.", 5),
    ("The classroom was not clean.", 2),
    ("The service was okay, nothing special.", 3),
    ("The staff were rude when I asked for help.", 1),
    ("The product works well and feels durable.", 5),
    ("The delivery was quick and the package was safe.", 5),
    ("The refund process was frustrating.", 1),
    ("The course content is useful.", 4),
    ("The application crashed while I was using it.", 1),
    ("The facilities need better maintenance.", 2),
    ("The price is reasonable and affordable.", 4),
    ("The customer care team solved my issue.", 5),
    ("The shipment tracking was not updated.", 2),
    ("The lecture was average.", 3),
    ("The product is defective and does not work.", 1),
    ("The website loads quickly and is simple.", 5),
    ("The library is clean and quiet.", 4),
    ("The staff responded quickly to my request.", 5),
    ("The quality is poor for the money spent.", 1),
    ("The course assignments are difficult to understand.", 2),
    ("The delivery was on time.", 4),
    ("The interface is confusing and slow.", 2),
]



# ---------- Negative Feedback Action Center helpers ----------
THEME_RESOLUTION_STEPS = {
    "Customer Service": [
        "Acknowledge the customer's experience and apologize for the inconvenience.",
        "Review the support conversation and identify where the response failed.",
        "Assign an owner and follow up with the customer after the issue is investigated.",
    ],
    "Delivery & Logistics": [
        "Check the order status, tracking history, and delivery partner records.",
        "Give the customer a clear update and a realistic delivery or resolution timeline.",
        "Review repeated delays with the delivery partner to prevent recurrence.",
    ],
    "Product Quality": [
        "Ask for the product model, order details, and a description of the defect if needed.",
        "Check warranty, repair, replacement, or refund options under the business policy.",
        "Record recurring defects and share them with the product or quality team.",
    ],
    "Pricing & Value": [
        "Clarify the price, included features, and any relevant charges without arguing with the customer.",
        "Check whether the product description and pricing information were clear.",
        "Share available support or return options according to the published policy.",
    ],
    "Usability & Experience": [
        "Ask which step, page, or feature caused difficulty.",
        "Reproduce the problem and record the device, browser, or error details if relevant.",
        "Prioritize a fix or provide a clear workaround and update the customer.",
    ],
    "Teaching & Learning": [
        "Ask for the specific class, topic, or learning difficulty.",
        "Review the material and offer clarification or additional learning support.",
        "Look for repeated concerns and discuss them with the appropriate academic team.",
    ],
    "Facilities & Cleanliness": [
        "Verify the reported location and inspect the facility.",
        "Arrange cleaning, repair, or maintenance if needed.",
        "Track completion and introduce a regular inspection checklist.",
    ],
    "Other / Unclassified": [
        "Read the full review carefully and identify the exact concern before choosing an action.",
        "Ask a respectful follow-up question if the complaint is unclear.",
        "Record the issue and route it to the team best placed to investigate it.",
    ],
}

def draft_public_reply(theme, feedback):
    """Create a safe, editable reply template; it does not contact the reviewer."""
    return (
        "Thank you for sharing your feedback. We're sorry to hear about your experience. "
        "We take concerns like this seriously and would like to understand the issue better. "
        "Please contact our support team with the relevant details so we can review this "
        "specific concern and discuss the available next steps. We appreciate you bringing "
        "this to our attention."
    )

def make_sample_csv():
    sample_df = pd.DataFrame(SAMPLE_REVIEWS, columns=["Review", "Rating"])
    return sample_df.to_csv(index=False).encode("utf-8-sig")


def read_uploaded_csv(uploaded_file):
    """Try common encodings and CSV separators; return a parsed dataframe."""
    last_error = None
    # utf-8-sig handles UTF-8 files with a BOM; cp1252 is common in Windows exports.
    # latin1 can decode any byte sequence, so we only use it after stricter options.
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        uploaded_file.seek(0)
        try:
            frame = pd.read_csv(uploaded_file, encoding=encoding, sep=None, engine="python")
            if len(frame.columns) == 1 and ";" in str(frame.columns[0]):
                uploaded_file.seek(0)
                frame = pd.read_csv(uploaded_file, encoding=encoding, sep=";")
            if len(frame.columns) == 1 and "\t" in str(frame.columns[0]):
                uploaded_file.seek(0)
                frame = pd.read_csv(uploaded_file, encoding=encoding, sep="\t")
            return frame
        except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
            last_error = exc
    raise ValueError(f"Could not read the CSV using common encodings. Last error: {last_error}")


def classify_sentiment(text):
    score = analyzer.polarity_scores(str(text))["compound"]
    if score >= 0.05:
        label = "Positive"
    elif score <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    return label, score


def detect_theme(text):
    text_lower = str(text).lower()
    matches = {}
    for theme, keywords in THEME_RULES.items():
        count = sum(
            1 for keyword in keywords
            if re.search(r"(?<!\w)" + re.escape(keyword) + r"(?!\w)", text_lower)
        )
        if count:
            matches[theme] = count
    if not matches:
        return "Other / Unclassified"
    return max(matches, key=matches.get)


def extract_keywords(texts, limit=15):
    if not texts:
        return pd.DataFrame(columns=["Keyword", "Frequency"])
    try:
        vectorizer = CountVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=limit,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
        )
        matrix = vectorizer.fit_transform(texts)
        return pd.DataFrame({
            "Keyword": vectorizer.get_feature_names_out(),
            "Frequency": matrix.sum(axis=0).A1,
        }).sort_values("Frequency", ascending=False).reset_index(drop=True)
    except ValueError:
        return pd.DataFrame(columns=["Keyword", "Frequency"])


def analyze_dataframe(frame, review_column):
    if review_column not in frame.columns:
        raise ValueError("Selected feedback column was not found.")
    original = frame[review_column].fillna("").astype(str).str.strip()
    valid = original.ne("")
    results = frame.loc[valid].copy()
    results["Feedback"] = original.loc[valid]
    if results.empty:
        raise ValueError("No non-empty feedback was found in the selected column.")
    predictions = results["Feedback"].apply(classify_sentiment)
    results["Sentiment"] = predictions.apply(lambda item: item[0])
    results["Sentiment Score"] = predictions.apply(lambda item: item[1])
    results["Theme"] = results["Feedback"].apply(detect_theme)
    return results.reset_index(drop=True), int((~valid).sum())


# ---------------- Navigation and dashboard ----------------
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "analysis_info" not in st.session_state:
    st.session_state.analysis_info = {}

with st.sidebar:
    st.markdown("## ◧ FeedbackLens AI")
    st.caption("Turn customer feedback into clear, practical improvements.")
    st.divider()
    page = st.radio(
        "Choose a section to explore your feedback:",
        ["Dashboard", "Insights", "Discover recurring feedback themes", "Reports"],
        label_visibility="visible",
        key="main_navigation",
    )
    st.divider()
    st.markdown("**Manage your feedback data**")
    st.caption("Upload a CSV file on the Dashboard to analyze customer comments and view the results here.")
    st.download_button(
        "Download a sample feedback dataset",
        data=make_sample_csv(),
        file_name="sample_feedback.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.markdown(
    '<div class="hero"><div class="eyebrow">Customer experience analytics</div>'
    '<h1>FeedbackLens AI</h1>'
    '<p>Understand customer feedback, identify recurring issues, and decide what to improve next.</p></div>',
    unsafe_allow_html=True,
)

# Dataset upload is intentionally shown only on the Dashboard page.
if page == "Dashboard":
    with st.expander(
        "Upload or change your feedback dataset",
        expanded=st.session_state.analysis_results is None,
    ):
        st.write(
            "Upload a CSV file containing customer reviews or feedback. "
            "Select the column with the full review text, then choose Analyze feedback."
        )
        uploaded = st.file_uploader("Choose CSV file", type=["csv"], key="feedback_csv_uploader")
        if uploaded is not None:
            try:
                raw_df = read_uploaded_csv(uploaded)
                if raw_df.empty:
                    st.error("The CSV file contains no rows.")
                elif len(raw_df) > 100000:
                    st.error("This app supports up to 100,000 rows per upload.")
                else:
                    text_columns = [
                        c for c in raw_df.columns
                        if pd.api.types.is_object_dtype(raw_df[c])
                        or pd.api.types.is_string_dtype(raw_df[c])
                    ]
                    if not text_columns:
                        st.error("No text column was found in this CSV.")
                    else:
                        def column_priority(column):
                            name = str(column).lower().replace("_", " ")
                            if any(term in name for term in (
                                "review text", "review body", "feedback text",
                                "full review", "comment text", "review content"
                            )):
                                return 0
                            if any(term in name for term in (
                                "review", "feedback", "comment", "response", "content", "text"
                            )) and "title" not in name:
                                return 1
                            if "title" in name or "headline" in name:
                                return 3
                            return 2

                        default_col = sorted(text_columns, key=column_priority)[0]
                        review_col = st.selectbox(
                            "Feedback text column",
                            text_columns,
                            index=text_columns.index(default_col),
                        )
                        st.caption(f"Dataset loaded: {len(raw_df):,} rows · {len(raw_df.columns)} columns")
                        if st.button("Analyze feedback", type="primary", use_container_width=True):
                            with st.spinner("Analyzing feedback…"):
                                analyzed, excluded = analyze_dataframe(raw_df, review_col)
                                st.session_state.analysis_results = analyzed
                                st.session_state.analysis_info = {
                                    "filename": uploaded.name,
                                    "excluded": excluded,
                                    "review_column": review_col,
                                    "original_rows": len(raw_df),
                                }
                            st.success("Analysis completed.")
            except Exception as exc:
                st.error(f"Could not read or analyze this CSV: {exc}")

results = st.session_state.analysis_results
info = st.session_state.analysis_info

if results is None:
    if page == "Dashboard":
        st.markdown("### Understand what your customers are telling you")
        st.info("To get started, upload a CSV file containing customer reviews. If you do not have a dataset yet, download the sample file from the sidebar to explore how the dashboard works.")
        demo1, demo2, demo3 = st.columns(3)
        with demo1:
            st.markdown(
                '<div class="insight-card"><div class="insight-title">Sentiment analysis</div>'
                '<div class="insight-body">Understand positive, neutral, and negative feedback at a glance.</div></div>',
                unsafe_allow_html=True,
            )
        with demo2:
            st.markdown(
                '<div class="insight-card"><div class="insight-title">Feedback clusters</div>'
                '<div class="insight-body">Identify recurring themes such as shipping, product quality, and support.</div></div>',
                unsafe_allow_html=True,
            )
        with demo3:
            st.markdown(
                '<div class="insight-card"><div class="insight-title">Recommended actions</div>'
                '<div class="insight-body">Review negative comments and plan practical next steps.</div></div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("Upload a feedback CSV from the Dashboard first. This section will appear after analysis.")

else:
    data = results.copy()
    total = len(data)
    sentiment_counts = data["Sentiment"].value_counts()
    pos = int(sentiment_counts.get("Positive", 0))
    neu = int(sentiment_counts.get("Neutral", 0))
    neg = int(sentiment_counts.get("Negative", 0))
    positive_pct = (pos / total * 100) if total else 0
    avg_score = float(data["Sentiment Score"].mean()) if total else 0

    if page == "Dashboard":
        st.caption(
            f"Dataset: {info.get('filename', 'Uploaded CSV')} · {total:,} feedback entries · "
            f"{info.get('excluded', 0):,} empty entries excluded"
        )
        metric_specs = [
            ("Total feedback received", f"{total:,}", f"{info.get('original_rows', total):,} rows were loaded from your dataset"),
            ("Customers with positive feedback", f"{positive_pct:.1f}%", f"{pos:,} reviews were classified as positive"),
            ("Average sentiment score", f"{avg_score:+.2f}", "Scores range from −1 for negative to +1 for positive"),
            ("Reviews that may need attention", f"{neg:,}", "These reviews were classified as negative and may need follow-up"),
        ]
        metric_cols = st.columns(4)
        for col, (label, value, note) in zip(metric_cols, metric_specs):
            with col:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">{label}</div>'
                    f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
                    unsafe_allow_html=True,
                )

        st.write("")
        chart_col, cluster_col = st.columns([1.8, 1])
        with chart_col:
            st.markdown("### How customers feel about their experience")
            # Pie chart for the overall positive, neutral, and negative split.
            sentiment_data = pd.DataFrame({
                "Sentiment": ["Positive", "Neutral", "Negative"],
                "Reviews": [pos, neu, neg],
            })
            sentiment_colors = {
                "Positive": "#a7df2e",
                "Neutral": "#a5b0c0",
                "Negative": "#fa6268",
            }
            fig = px.pie(
                sentiment_data,
                names="Sentiment",
                values="Reviews",
                hole=0.48,
                color="Sentiment",
                color_discrete_map=sentiment_colors,
                title="How feedback is distributed across sentiment categories",
            )
            fig.update_traces(
                textposition="inside",
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>Reviews: %{value}<br>Share: %{percent}<extra></extra>",
                marker=dict(line=dict(color="#ffffff", width=3)),
            )
            fig.update_layout(
                height=330,
                paper_bgcolor="white",
                plot_bgcolor="white",
                font=dict(color="#526174", size=12),
                margin=dict(t=55, b=15, l=15, r=15),
                legend_title_text="",
                showlegend=True,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("The chart shows the number and percentage of positive, neutral, and negative reviews in the dataset you analyzed.")

        with cluster_col:
            st.markdown("### Topics customers mention most often")
            cluster_counts = data["Theme"].value_counts().head(7)
            palette_list = ["#b4e63d", "#ffbf3f", "#8bd94b", "#4c8bf5", "#9b6cff", "#f36f91", "#aab4c0"]
            for idx, (theme, count) in enumerate(cluster_counts.items()):
                pct = count / total * 100 if total else 0
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:9px;padding:9px 0;border-bottom:1px solid #edf0f3;">'
                    f'<div style="width:4px;height:25px;border-radius:4px;background:{palette_list[idx % len(palette_list)]};"></div>'
                    f'<div style="flex:1;font-size:12px;color:#4b596a;font-weight:600;">{theme}</div>'
                    f'<div style="font-size:11px;color:#8a96a5;">{pct:.0f}%</div></div>',
                    unsafe_allow_html=True,
                )

        st.write("")
        recent_col, insight_col = st.columns([1.55, 1])
        with recent_col:
            st.markdown("### What customers have said recently")
            for _, row in data.tail(5).iloc[::-1].iterrows():
                sentiment = str(row["Sentiment"])
                accent = {"Positive": "#9bd832", "Negative": "#ff6868", "Neutral": "#f2bd45"}.get(sentiment, "#aab4c0")
                snippet = str(row["Feedback"])
                st.markdown(
                    f'<div style="background:#fff;border:1px solid #e9edf2;border-left:3px solid {accent};'
                    f'border-radius:9px;padding:11px 13px;margin-bottom:8px;">'
                    f'<div style="display:flex;justify-content:space-between;gap:10px;">'
                    f'<b style="font-size:12px;color:#344256;">{sentiment}</b>'
                    f'<span style="font-size:10px;color:#8a96a5;">{row["Theme"]}</span></div>'
                    f'<div style="font-size:12px;color:#647184;line-height:1.5;margin-top:5px;">{snippet[:240]}</div></div>',
                    unsafe_allow_html=True,
                )
        with insight_col:
            st.markdown("### Insights and recommended next steps")
            negative_rows = data[data["Sentiment"] == "Negative"]
            if not negative_rows.empty:
                theme_counts = negative_rows["Theme"].value_counts()
                top_theme = theme_counts.index[0]
                st.markdown(
                    f'<div class="insight-card"><div class="insight-title">{top_theme}</div>'
                    f'<div class="insight-body">{int(theme_counts.iloc[0])} negative review(s). '
                    f'{THEME_RECOMMENDATIONS.get(top_theme, THEME_RECOMMENDATIONS["Other / Unclassified"])}</div></div>',
                    unsafe_allow_html=True,
                )
                words = extract_keywords(negative_rows["Feedback"].tolist(), limit=8)
                if not words.empty:
                    st.markdown(
                        '<div class="insight-card"><div class="insight-title">Recurring negative keywords</div>'
                        '<div class="insight-body">' + " · ".join(words["Keyword"].head(5).astype(str).tolist())
                        + "</div></div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    '<div class="insight-card"><div class="insight-title">No negative reviews detected</div>'
                    '<div class="insight-body">Continue reviewing new feedback as it arrives.</div></div>',
                    unsafe_allow_html=True,
                )

    elif page == "Insights":
        st.markdown("## Customer feedback insights")
        st.caption("Explore frequently mentioned words, review negative comments, and use the suggested actions to understand what may need attention.")
        keywords = extract_keywords(data["Feedback"].tolist(), limit=20)
        if not keywords.empty:
            left, right = st.columns([1.4, 1])
            with left:
                fig = px.bar(
                    keywords.head(12).sort_values("Frequency"),
                    x="Frequency",
                    y="Keyword",
                    orientation="h",
                    color="Frequency",
                    color_continuous_scale=["#d8efa0", "#91c51d"],
                    title="Frequently used words and phrases",
                )
                fig.update_layout(
                    height=400,
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#ffffff",
                    font=dict(color="#344256", size=13),
                    title_font=dict(color="#26364a", size=16),
                    coloraxis_showscale=False,
                    xaxis=dict(title_font=dict(color="#344256"), tickfont=dict(color="#526174"), gridcolor="#e8edf2"),
                    yaxis=dict(title_font=dict(color="#344256"), tickfont=dict(color="#526174"), gridcolor="#ffffff"),
                    margin=dict(l=10, r=15, t=55, b=35),
                )
                st.plotly_chart(fig, use_container_width=True)
            with right:
                st.markdown("### Words and phrases mentioned in customer comments")
                st.dataframe(keywords, use_container_width=True, hide_index=True)
        negative_rows = data[data["Sentiment"] == "Negative"]
        st.markdown("### Review a comment that may need attention")
        if not negative_rows.empty:
            selected = st.selectbox("Choose a review", negative_rows["Feedback"].tolist())
            theme = detect_theme(selected)
            st.info(selected)
            st.markdown("**Suggested next steps**")
            for step in THEME_RESOLUTION_STEPS.get(theme, THEME_RESOLUTION_STEPS["Other / Unclassified"]):
                st.markdown(f"- {step}")
            st.caption("Review and edit the suggested response below before using it. The app will not send this message to the customer.")
            st.text_area(
                "Edit your response to the customer",
                value=draft_public_reply(theme, selected),
                height=150,
                key="editable_public_reply_draft",
                help="Click inside this box to rewrite the response in your own words. Your edits remain in the app and are not sent automatically.",
            )
        else:
            st.info("There are no negative reviews in this dataset.")

    elif page == "Discover recurring feedback themes":
        st.markdown("## Explore feedback by topic")
        st.caption("Explore the topics customers mention most often and compare the positive, neutral, and negative feedback within each topic.")
        cluster_table = data.groupby(["Theme", "Sentiment"]).size().reset_index(name="Feedback count")
        st.dataframe(cluster_table.sort_values("Feedback count", ascending=False), use_container_width=True, hide_index=True)
        selected_theme = st.selectbox("Choose a feedback cluster", sorted(data["Theme"].unique()))
        st.markdown("### What you can do about this topic")
        st.write(THEME_RECOMMENDATIONS.get(selected_theme, THEME_RECOMMENDATIONS["Other / Unclassified"]))
        st.markdown("### Customer comments related to this topic")
        st.dataframe(
            data[data["Theme"] == selected_theme][["Feedback", "Sentiment", "Theme"]],
            use_container_width=True,
            hide_index=True,
        )

    elif page == "Reports":
        st.markdown("## Download your feedback reports")
        st.caption("Export the analyzed feedback for further review or download a written summary of the sentiment results and recurring themes.")
        report = (
            f"FEEDBACKLENS AI — FEEDBACK REPORT\n"
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"Dataset: {info.get('filename', 'Uploaded CSV')}\n\n"
            f"Total feedback: {total}\nPositive: {pos} ({pos/total:.1%})\n"
            f"Neutral: {neu} ({neu/total:.1%})\nNegative: {neg} ({neg/total:.1%})\n"
            f"Average sentiment score: {avg_score:+.2f}\n\n"
            f"Feedback clusters:\n{data['Theme'].value_counts().to_string()}\n\n"
            "Sentiment and themes are automated estimates. Review original comments before making decisions."
        )
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Download analyzed CSV",
                data=data.to_csv(index=False).encode("utf-8-sig"),
                file_name="feedbacklens_analyzed.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col2:
            st.download_button(
                "Download summary report",
                data=report,
                file_name="feedbacklens_report.txt",
                mime="text/plain",
                use_container_width=True,
            )
        st.markdown("### Summary of customer sentiment")
        summary = pd.DataFrame({
            "Sentiment": ["Positive", "Neutral", "Negative"],
            "Reviews": [pos, neu, neg],
            "Share": [f"{pos/total:.1%}", f"{neu/total:.1%}", f"{neg/total:.1%}"],
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

st.divider()
st.caption("FeedbackLens AI provides automated sentiment labels and suggested actions. Please review the original customer comments before making business decisions.")
