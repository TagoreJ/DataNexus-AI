import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from pandasai import Agent
from pandasai.llm import OpenAI
from datetime import datetime
import io
import json
from scipy import stats
from scipy.stats import chi2_contingency, spearmanr, kendalltau
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, davies_bouldin_score
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings("ignore")

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
    PageBreak, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

try:
    import pygwalker as pyg
except ImportError:
    pyg = None

# ================================================
# CUSTOM HTML/CSS STYLING
# ================================================

CUSTOM_HTML_CSS = """
<style>
    :root {
        --primary: #1f77b4;
        --secondary: #ff7f0e;
        --success: #2ca02c;
        --danger: #d62728;
        --info: #17a2b8;
        --light: #f8f9fa;
        --dark: #212529;
    }
    
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        margin: 10px 0;
    }
    
    .insight-box {
        background: #f0f7ff;
        border-left: 4px solid #1f77b4;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        color: #1a1a1a; 
    }
    
    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        color: #664d03
    }
    
    .success-box {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .categorical-box {
        background: #e7f3ff;
        border-left: 4px solid #0066cc;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        color: #003d99;
    }
    
    .numeric-box {
        background: #f0f8ff;
        border-left: 4px solid #667eea;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        color: #1a1a4d;
    }
    
    .header-gradient {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 15px;
        margin: 20px 0;
    }
    
    .code-block {
        background: #f5f5f5;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 15px;
        font-family: 'Courier New', monospace;
        overflow-x: auto;
    }
    
    .data-type-badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 12px;
        font-weight: bold;
        margin: 2px;
    }
    
    .numeric-badge {
        background: #667eea;
        color: white;
    }
    
    .categorical-badge {
        background: #ff7f0e;
        color: white;
    }
    
    .pro-feature {
        background: #fff8e1;
        border: 2px solid #ffd700;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        color: #856404;
        font-weight: bold;
    }
</style>
"""


# ================================================
# PAGE CONFIG & THEME
# ================================================

st.set_page_config(
    page_title="DataNexus AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(CUSTOM_HTML_CSS, unsafe_allow_html=True)


# ================================================
# HEADER WITH GRADIENT
# ================================================

st.markdown("""
<div class="header-gradient">
    <h1>🧬 DataNexus AI Ultimate</h1>
    <p style="font-size: 16px; margin: 10px 0;">Enterprise-Grade Analytics • Neural Agent • ML & Statistics • Reports</p>
    <p style="font-size: 12px; opacity: 0.8;">Powered by Advanced Statistics & Large Language Model Reasoning</p>
</div>
""", unsafe_allow_html=True)

# ================================================
# LLM ENGINE (PandasAI Integration)
# ================================================
class OpenRouterLLM:
    """Adapter for OpenRouter API to work with PandasAI."""
    
    MODELS = {
        
        "⚙️ MiMo-V2-Flash (Lightweight)": "xiaomi/mimo-v2-flash:free",
        "📊 Trinity Mini (Fast Analysis)": "arcee-ai/trinity-mini:free",
        "🦾 Llama 3.3 70B (Balanced Power)": "meta-llama/llama-3.3-70b-instruct:free",
        "🧩 Nemotron 3 Nano (Ultra-Fast)": "nvidia/nemotron-3-nano-30b-a3b:free",
        "🧠 DeepSeek R1 0528 (Reasoning)": "deepseek/deepseek-r1-0528:free",
    }

    @staticmethod
    @st.cache_resource
    def get_llm(api_key: str, model_name: str):
        if not api_key:
            return None
        # Initialize with a supported model to bypass validation
        llm = OpenAI(
            api_token=api_key,
            api_base="https://openrouter.ai/api/v1",
            model="gpt-3.5-turbo",
            temperature=0.1
        )
        # Override with the actual OpenRouter model
        llm.model = OpenRouterLLM.MODELS[model_name]
        return llm

def get_agent(df, api_key, model_name, enforce_privacy=True, enable_cache=True):
    """Factory for PandasAI Agent with correct LLM setup."""
    if not api_key:
        return None
    
    llm = OpenRouterLLM.get_llm(api_key, model_name)
    agent = Agent(
        df,
        config={
            "llm": llm,
            "save_charts": True,
            "save_charts_path": "./charts",
            "enforce_privacy": enforce_privacy,
            "enable_cache": enable_cache,
            "verbose": True
        }
    )
    return agent

# ================================================
# UTILITY FUNCTIONS
# ================================================

def handle_missing_values(df, numeric_strat="Median", categorical_strat="Most Frequent"):
    """Handle missing values separately for numeric and categorical columns"""
    if df is None or not isinstance(df, pd.DataFrame):
        return None
    if df.empty:
        return df.copy()
    
    df_filled = df.copy()
    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    
    # Numeric columns imputation
    if num_cols:
        if numeric_strat == "Median":
            num_imputer = SimpleImputer(strategy="median")
            df_filled[num_cols] = num_imputer.fit_transform(df[num_cols])
        elif numeric_strat == "Mean":
            num_imputer = SimpleImputer(strategy="mean")
            df_filled[num_cols] = num_imputer.fit_transform(df[num_cols])
        elif numeric_strat == "Drop Rows":
            df_filled = df_filled.dropna(subset=num_cols)
    
    # Categorical columns imputation
    if cat_cols:
        if categorical_strat == "Most Frequent":
            cat_imputer = SimpleImputer(strategy="most_frequent")
            df_filled[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
        elif categorical_strat == "Missing Label":
            df_filled[cat_cols] = df_filled[cat_cols].fillna("Missing")
        elif categorical_strat == "Drop Rows":
            df_filled = df_filled.dropna(subset=cat_cols)
    
    return df_filled


def calculate_advanced_statistics(df):
    """Calculate comprehensive statistics for both numeric and categorical data"""
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Invalid DataFrame passed to calculate_advanced_statistics")
    if df.empty:
        return {
            "shape": df.shape,
            "memory": 0,
            "numeric_cols": [],
            "categorical_cols": [],
            "missing_pct": {},
            "duplicates": 0,
            "numeric_stats": {},
            "categorical_stats": {},
        }
    
    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    
    stats_dict = {
        "shape": df.shape,
        "memory": df.memory_usage(deep=True).sum() / 1024**2,
        "numeric_cols": num_cols,
        "categorical_cols": cat_cols,
        "missing_pct": (df.isnull().sum() / len(df) * 100).to_dict(),
        "duplicates": df.duplicated().sum(),
        "numeric_stats": df[num_cols].describe().to_dict() if num_cols else {},
        "categorical_stats": {col: df[col].value_counts().to_dict() for col in cat_cols},
    }
    return stats_dict


def detect_anomalies(df, numeric_cols, method="iqr"):
    """Detect anomalies using IQR or Isolation Forest"""
    if not numeric_cols or df is None:
        return {}
    
    anomalies = {}
    
    if method == "iqr":
        for col in numeric_cols:
            try:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower) | (df[col] > upper)][col].tolist()
                anomalies[col] = {
                    "count": len(outliers),
                    "percentage": len(outliers) / len(df) * 100 if len(df) > 0 else 0,
                    "bounds": (lower, upper)
                }
            except:
                pass
    
    elif method == "isolation_forest":
        try:
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_labels = iso_forest.fit_predict(df[numeric_cols])
            anomalies["isolation_forest"] = {
                "count": (anomaly_labels == -1).sum(),
                "percentage": (anomaly_labels == -1).sum() / len(df) * 100
            }
        except:
            pass
    
    return anomalies


def analyze_categorical(df, cat_cols):
    """Analyze categorical variables comprehensively"""
    if not cat_cols or df is None:
        return {}
    
    cat_analysis = {}
    for col in cat_cols:
        try:
            value_counts = df[col].value_counts()
            cat_analysis[col] = {
                "unique": df[col].nunique(),
                "mode": df[col].mode()[0] if len(df[col].mode()) > 0 else "N/A",
                "missing": df[col].isnull().sum(),
                "top_3": value_counts.head(3).to_dict(),
                "cardinality": (df[col].nunique() / len(df)) * 100 if len(df) > 0 else 0,
                "is_imbalanced": value_counts.iloc[0] / value_counts.sum() > 0.8 if len(value_counts) > 0 else False
            }
        except:
            pass
    
    return cat_analysis


def calculate_vif(df, numeric_cols):
    """Calculate Variance Inflation Factor for multicollinearity"""
    if not numeric_cols or len(numeric_cols) < 2 or df is None:
        return None
    
    try:
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df[numeric_cols].dropna())
        df_scaled = pd.DataFrame(df_scaled, columns=numeric_cols)
        
        vif_data = pd.DataFrame()
        vif_data["Feature"] = numeric_cols
        vif_data["VIF"] = [variance_inflation_factor(df_scaled.values, i) for i in range(len(numeric_cols))]
        return vif_data.sort_values("VIF", ascending=False)
    except:
        return None

def plotly_to_bytes(fig):
    """Convert plotly figure to PNG bytes"""
    try:
        return fig.to_image(format="png", width=1000, height=600)
    except:
        return None


def create_professional_pdf(df, title, insights, charts_bytes):
    """Create professional PDF report with ReportLab"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    story = []
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=28,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=10,
        alignment=TA_CENTER
    )
    
    # Title
    story.append(Paragraph(title, title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M %p')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Insights
    if insights:
        story.append(Paragraph("Key Findings & Insights", styles['Heading2']))
        for insight in insights[:8]:
            try:
                story.append(Paragraph(f"• {str(insight)[:250]}", styles['Normal']))
            except:
                pass
        story.append(Spacer(1, 20))
    
    # Data Preview Table
    if df is not None and not df.empty:
        story.append(Paragraph("Data Overview (First 10 Rows)", styles['Heading2']))
        try:
            cols_to_show = df.columns.tolist()[:5]
            data = [cols_to_show] + [list(df[cols_to_show].iloc[i]) for i in range(min(10, len(df)))]
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0f0f0')),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ]))
            story.append(table)
            story.append(PageBreak())
        except:
            pass
    
    # Charts
    for i, img_bytes in enumerate(charts_bytes):
        if img_bytes:
            try:
                story.append(Paragraph(f"Visualization {i+1}", styles['Heading2']))
                img = Image(io.BytesIO(img_bytes), width=6*inch, height=4*inch)
                story.append(img)
                story.append(Spacer(1, 20))
                if i % 2 == 1:
                    story.append(PageBreak())
            except:
                pass
    
    doc.build(story)
    return buffer.getvalue()


# ================================================
# SIDEBAR CONFIGURATION
# ================================================

with st.sidebar:
    st.markdown("### ⚙️ Engine Configuration")
    
    # API Key
    api_key = st.text_input(
        "🔑 OpenRouter API Key",
        type="password",
        help="Get free key at https://openrouter.ai/keys"
    )
    
    # Model Selection
    st.markdown("### 🤖 AI Model Selection")
    model_display = st.selectbox("Select Reasoning Engine", list(OpenRouterLLM.MODELS.keys()))
    
    # Analysis Settings
    with st.expander("🔧 Advanced Calibration"):
        confidence_level = st.slider("Confidence Level", 0.80, 0.99, 0.95, 0.01)
        correlation_threshold = st.slider("Correlation Threshold", 0.0, 1.0, 0.5, 0.1)
        
        # Missing Value Strategy
        st.markdown("**Missing Data Strategy**")
        numeric_strategy = st.selectbox(
            "Numeric Imputation",
            ["Median", "Mean", "Drop Rows"],
        )
        categorical_strategy = st.selectbox(
            "Categorical Imputation",
            ["Most Frequent", "Missing Label", "Drop Rows"],
        )
        
        enable_anomaly = st.checkbox("Enable Anomaly Detection", value=True)
        enable_vif = st.checkbox("Enable Multicollinearity Check", value=False)
        enforce_privacy = st.toggle("Enforce Privacy", value=True)
        enable_cache = st.toggle("Enable Caching", value=True)
    
    # Session Controls
    st.markdown("---")
    if st.button("🔄 Reset Session", type="secondary"):
        st.session_state.clear()
        st.rerun()

# ================================================
# SESSION STATE INITIALIZATION
# ================================================

if "df" not in st.session_state:
    st.session_state.df = None
if "df_cleaned" not in st.session_state:
    st.session_state.df_cleaned = None
if "df_stats" not in st.session_state:
    st.session_state.df_stats = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ================================================
# MAIN LAYOUT & TABS
# ================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📥 Upload & Explore",
    "📊 A.I. Dashboard",
    "🧠 Neural Analyst",
    "🔍 Deep Analysis",
    "📈 Statistical Tests",
    "🤖 Machine Learning",
    "🎨 Visual Explorer",
    "🎯 Pattern Mining",
    "📄 Reports"
])


# ================================================
# TAB 1: UPLOAD & EXPLORE
# ================================================

with tab1:
    st.header("📥 Data Upload & Exploration")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded = st.file_uploader(
            "📁 Drag and drop CSV, Excel, or JSON file",
            type=["csv", "xlsx", "xls", "json"],
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        sample_data = st.button("📋 Load Sample Data (Demo)")
    
    if sample_data:
        df = pd.DataFrame({
            'ID': range(1, 101),
            'Sales': np.random.randint(1000, 10000, 100),
            'Profit': np.random.randint(100, 2000, 100),
            'Region': np.random.choice(['North', 'South', 'East', 'West'], 100),
            'Category': np.random.choice(['Electronics', 'Clothing', 'Food'], 100),
            'Date': pd.date_range('2024-01-01', periods=100, freq='D'),
            'Customer_Satisfaction': np.random.uniform(1, 5, 100),
            'Units_Sold': np.random.randint(1, 100, 100)
        })
        st.session_state.df = df
        st.session_state.df_cleaned = handle_missing_values(df, numeric_strategy, categorical_strategy)
        st.session_state.df_stats = calculate_advanced_statistics(st.session_state.df_cleaned)
        st.success("✅ Sample data loaded successfully!")
    
    elif uploaded:
        try:
            if uploaded.name.endswith('.csv'):
                df = pd.read_csv(uploaded)
            elif uploaded.name.endswith('.json'):
                df = pd.read_json(uploaded)
            else:
                df = pd.read_excel(uploaded, engine="openpyxl")
                df.columns = df.columns.astype(str)
            
            st.session_state.df = df
            st.session_state.df_cleaned = handle_missing_values(df, numeric_strategy, categorical_strategy)
            st.session_state.df_stats = calculate_advanced_statistics(st.session_state.df_cleaned)
            st.success(f"✅ Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
        except Exception as e:
            st.error(f"❌ Error: {e}")
    
    # Display statistics
    if st.session_state.df is not None:
        stats_data = st.session_state.df_stats
        
        st.markdown("### 📈 Dataset Statistics")
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1: st.metric("📊 Rows", f"{stats_data['shape'][0]:,}")
        with m2: st.metric("📋 Columns", stats_data['shape'][1])
        with m3: st.metric("💾 Size (MB)", f"{stats_data['memory']:.2f}")
        with m4:
            missing_count = sum(1 for v in stats_data['missing_pct'].values() if v > 0)
            st.metric("❌ Missing Cols", missing_count)
        with m5: st.metric("🔄 Duplicates", stats_data['duplicates'])
        
        st.markdown("### 👀 Data Preview")
        st.dataframe(st.session_state.df.head(20), use_container_width=True)

# ================================================
# TAB 2: A.I. DASHBOARD
# ================================================

with tab2:
    st.header("📊 A.I. Executive Dashboard")
    
    if st.session_state.df_cleaned is None:
        st.info("👈 Please load data first.")
    else:
        df = st.session_state.df_cleaned
        
        # --- AI Auto-Generated Section ---
        st.markdown("""
        <div class="insight-box">
            <strong>🤖 AI Insight Generator</strong><br>
            Instantly generate an executive summary and key visualization using the LLM agent.
        </div>
        """, unsafe_allow_html=True)
        
        col_ctrl, col_display = st.columns([1, 4])
        
        with col_ctrl:
            generate_btn = st.button("🚀 Generate Insights", type="primary", use_container_width=True)
        
        if generate_btn:
            if not api_key:
                st.warning("⚠️ Please provide an API Key in the sidebar.")
            else:
                agent = get_agent(df, api_key, model_display, enforce_privacy, enable_cache)
                
                col_ai_1, col_ai_2 = st.columns(2)
                
                with col_ai_1:
                    st.markdown("### 📋 Executive Summary")
                    with st.status("Drafting Executive Summary...", expanded=True) as status:
                        try:
                            st.write("🔍 Analyzing patterns...")
                            summary = agent.chat("Analyze this dataset and provide a 3-bullet executive summary of key trends, anomalies, or important metrics. Be concise.")
                            st.markdown(summary)
                            status.update(label="Analysis Complete", state="complete", expanded=True)
                        except Exception as e:
                            st.error(f"Error: {e}")
                            status.update(label="Failed", state="error")
                
                with col_ai_2:
                    st.markdown("### 📊 Key Visualization")
                    with st.status("Creating Visualization...", expanded=True) as status:
                        try:
                            st.write("🎨 Plotting trends...")
                            chart_response = agent.chat("Generate the single most important chart for this data (e.g., trends or distribution). Return the plot image path.")
                            if isinstance(chart_response, str) and (chart_response.endswith('.png') or chart_response.endswith('.jpg')):
                                st.image(chart_response)
                                status.update(label="Chart Generated", state="complete", expanded=True)
                            else:
                                st.write(chart_response)
                                status.update(label="Insight Generated", state="complete", expanded=True)
                        except Exception as e:
                            st.error(f"Error: {e}")
                            status.update(label="Failed", state="error")
        
        st.markdown("---")
        st.markdown("### 🛠️ Manual Dashboard Controls")
        
        # Manual Dashboard Logic (from previous user script)
        num_cols = df.select_dtypes(include='number').columns.tolist()
        cat_cols = df.select_dtypes(include='object').columns.tolist()
        
        analysis_type = st.radio("Select View", ["Numeric Analysis", "Categorical Analysis", "Mixed Analysis"], horizontal=True)
        
        if analysis_type == "Numeric Analysis" and num_cols:
            c1, c2, c3 = st.columns(3)
            with c1: chart_type = st.selectbox("Chart", ["Distribution", "Scatter", "Box Plot", "Heatmap"])
            with c2: x_col = st.selectbox("X Column", num_cols)
            with c3: y_col = st.selectbox("Y Column", [c for c in num_cols if c != x_col]) if len(num_cols) > 1 else num_cols[0]
            
            if chart_type == "Distribution":
                fig = px.histogram(df, x=x_col, nbins=30, color_discrete_sequence=['#667eea'])
                st.plotly_chart(fig, use_container_width=True)
            elif chart_type == "Scatter":
                fig = px.scatter(df, x=x_col, y=y_col)
                st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Categorical Analysis" and cat_cols:
            c1, c2 = st.columns(2)
            with c1: cat_col = st.selectbox("Category", cat_cols)
            with c2: fig = px.bar(df[cat_col].value_counts().reset_index(), x=cat_col, y='count')
            st.plotly_chart(fig, use_container_width=True)


# ================================================
# TAB 3: NEURAL ANALYST (PandasAI AGENT)
# ================================================

with tab3:
    st.header("🧠 Neural Analyst")
    st.info("💬 Chat with your data using powerful AI agents. Supports pivoting, plotting, and reasoning.")
    
    if st.session_state.df_cleaned is None:
        st.info("👈 Please load data first.")
    elif not api_key:
        st.warning("⚠️ Enter OpenRouter API Key in Sidebar")
    else:
        df = st.session_state.df_cleaned
        
        # Initialize Agent
        agent = get_agent(df, api_key, model_display, enforce_privacy, enable_cache)
        
        # Chat History
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg.get("type") == "image":
                    st.image(msg["content"])
                else:
                    st.markdown(msg["content"])
        
        # Input
        if prompt := st.chat_input("Ask: 'Plot sales per region' or 'Pivot table of profit by category'"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                response = None
                with st.status("🧠 Neural Engine processing...", expanded=True) as status:
                    st.write("⏳ Analyzing query...")
                    try:
                        response = agent.chat(prompt)
                        status.update(label="Processing Complete", state="complete", expanded=False)
                    except Exception as e:
                        st.error(f"Analysis Error: {e}")
                        status.update(label="Error", state="error")
                
                if response is not None:
                    # Handle IO types
                    if isinstance(response, str) and (response.endswith('.png') or response.endswith('.jpg')):
                        st.image(response)
                        st.session_state.messages.append({"role": "assistant", "content": response, "type": "image"})
                    
                    elif isinstance(response, pd.DataFrame):
                        st.dataframe(response)
                        st.session_state.messages.append({"role": "assistant", "content": response.to_markdown(), "type": "text"})
                        st.download_button(
                            "📥 Download CSV",
                            data=response.to_csv(index=False).encode('utf-8'),
                            file_name='ai_result.csv',
                            mime='text/csv'
                        )
                    
                    else:
                        st.write(response)
                        st.session_state.messages.append({"role": "assistant", "content": str(response), "type": "text"})
                    
                    # Show Code
                    if agent.last_code_generated:
                        with st.expander("📝 Show Generated Python Code"):
                            st.code(agent.last_code_generated, language='python')


# ================================================
# TAB 4: DEEP ANALYSIS
# ================================================

with tab4:
    st.header("🔍 Deep Statistical Analysis")
    
    if st.session_state.df_cleaned is not None:
        df = st.session_state.df_cleaned
        num_cols = df.select_dtypes(include='number').columns.tolist()
        cat_cols = df.select_dtypes(include='object').columns.tolist()
        
        subtab1, subtab2, subtab3 = st.tabs(["Anomaly Detection", "Categorical breakdown", "Correlations"])
        
        with subtab1:
            if num_cols and enable_anomaly:
                st.markdown("### 🚨 Anomaly Detection (IQR)")
                anomalies = detect_anomalies(df, num_cols, method="iqr")
                for col, anom in anomalies.items():
                    if anom['count'] > 0:
                        st.warning(f"**{col}**: {anom['count']} outliers detected ({anom['percentage']:.2f}%)")
            else:
                st.info("Enable Anomaly Detection in Sidebar or ensure numeric columns exist.")
        
        with subtab2:
            if cat_cols:
                st.markdown("### 🏷️ Categorical Insights")
                cat_analysis = analyze_categorical(df, cat_cols)
                for col, data in cat_analysis.items():
                    with st.expander(f"Analysis: {col}"):
                        st.json(data)
        
        with subtab3:
            if len(num_cols) > 1:
                st.markdown("### 🔗 Correlation Matrix")
                corr = df[num_cols].corr()
                fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu", zmin=-1, zmax=1)
                st.plotly_chart(fig, use_container_width=True)

# ================================================
# TAB 5: STATISTICAL TESTS
# ================================================

with tab5:
    st.header("📈 Statistical Tests")
    
    if st.session_state.df_cleaned is not None:
        df = st.session_state.df_cleaned
        num_cols = df.select_dtypes(include='number').columns.tolist()
        
        test_type = st.radio("Choose Test", ["Normality (Shapiro-Wilk)", "Pearson Correlation"])
        
        if test_type == "Normality (Shapiro-Wilk)" and num_cols:
            col_sel = st.selectbox("Column", num_cols, key="shapiro_col")
            if st.button("Run Test"):
                try:
                    stat, p = stats.shapiro(df[col_sel].dropna())
                    st.metric("P-Value", f"{p:.5f}")
                    if p > 0.05: st.success("Data is likely Normal")
                    else: st.warning("Data is likely NOT Normal")
                except Exception as e: st.error(str(e))


# ================================================
# TAB 6: MACHINE LEARNING
# ================================================

with tab6:
    st.header("🤖 Machine Learning")
    
    if st.session_state.df_cleaned is not None:
        df = st.session_state.df_cleaned
        num_cols = df.select_dtypes(include='number').columns.tolist()
        
        if len(num_cols) >= 2:
            st.markdown("### K-Means Clustering")
            k = st.slider("Number of Clusters", 2, 8, 3)
            if st.button("Run Clustering"):
                scaler = StandardScaler()
                scaled = scaler.fit_transform(df[num_cols])
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(scaled)
                
                df_clustered = df.copy()
                df_clustered['Cluster'] = clusters
                
                fig = px.scatter(df_clustered, x=num_cols[0], y=num_cols[1], color='Cluster')
                st.plotly_chart(fig, use_container_width=True)


# ================================================
# TAB 7: VISUAL EXPLORER (PyGWalker)
# ================================================

with tab7:
    st.header("🎨 Visual Explorer")
    if pyg is None:
        st.warning("PyGWalker not installed.")
    elif st.session_state.df_cleaned is not None:
        st.info("💡 Drag and drop fields to build Tableau-style charts.")
        pyg_html = pyg.to_html(st.session_state.df_cleaned)
        st.components.v1.html(pyg_html, height=800, scrolling=True)
    else:
        st.info("👈 Load data first.")


# ================================================
# TAB 8: PATTERN MINING
# ================================================
with tab8:
    st.header("🎯 Pattern Mining")
    if st.session_state.df_cleaned is not None:
        df = st.session_state.df_cleaned
        num_cols = df.select_dtypes(include='number').columns.tolist()
        if len(num_cols) > 1:
            st.markdown("**High Correlation Pairs (>0.5)**")
            corr = df[num_cols].corr()
            pairs = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    val = corr.iloc[i, j]
                    if abs(val) > 0.5:
                        pairs.append({"V1": corr.columns[i], "V2": corr.columns[j], "Correlation": val})
            if pairs:
                st.dataframe(pd.DataFrame(pairs))
            else:
                st.info("No strong correlations found.")


# ================================================
# TAB 9: REPORTS
# ================================================

with tab9:
    st.header("📄 PDF Reports")
    
    if st.session_state.df_cleaned is not None:
        report_title = st.text_input("Report Title", "DataNexus Analysis Report")
        if st.button("🚀 Generate PDF Report"):
            with st.spinner("Compiling report..."):
                df = st.session_state.df_cleaned
                charts = []
                try:
                    num_cols = df.select_dtypes(include='number').columns.tolist()
                    if num_cols:
                        fig = px.histogram(df, x=num_cols[0], title="Distribution Overview")
                        charts.append(plotly_to_bytes(fig))
                except: pass
                
                insights = [
                    f"Rows: {df.shape[0]}, Cols: {df.shape[1]}",
                    f"Duplicates: {df.duplicated().sum()}",
                ]
                
                try:
                    pdf_data = create_professional_pdf(df, report_title, insights, charts)
                    st.download_button("📥 Download PDF", data=pdf_data, file_name=f"{report_title}.pdf", mime="application/pdf")
                    st.success("Report generated!")
                except Exception as e:
                    st.error(f"Failed to generate PDF: {e}") 