import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
from pandasai.responses.response_parser import ResponseParser
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import pygwalker as pyg
except ImportError:
    pyg = None
import io
import time
import json

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="DataNexus AI | Pro Analytics",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom CSS/JS
def inject_custom_ui():
    st.markdown("""
    <style>
        :root {
            --primary-color: #00D4FF;
            --bg-dark: #0E1117;
            --glass-bg: rgba(26, 28, 36, 0.95);
            --border-glow: 1px solid rgba(0, 212, 255, 0.3);
        }
        /* Modern Container Styling */
        .stApp {
            background-color: var(--bg-dark);
        }
        div[data-testid="stSidebar"] {
            background-color: #111;
            border-right: 1px solid #333;
        }
        /* Custom Cards */
        .metric-card {
            background: var(--glass-bg);
            border: var(--border-glow);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            transition: transform 0.2s;
        }
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0, 212, 255, 0.1);
        }
        /* Chat Message Styling */
        .stChatMessage {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            border: 1px solid #333;
        }
        /* PyGWalker Container */
        .pygwalker-container {
            margin-top: 20px;
            border-radius: 12px;
            overflow: hidden;
        }
    </style>
    """, unsafe_allow_html=True)

inject_custom_ui()

# --- 2. LLM ENGINE (Strategy Pattern) ---
class OpenRouterLLM:
    """Adapter for OpenRouter API to work with PandasAI."""
    
    MODELS = {
        "Gemini 2.0 Flash (Fast & Free)": "google/gemini-2.0-flash-exp:free",
        "Llama 3.3 70B (Meta SOTA)": "meta-llama/llama-3.3-70b-instruct:free",
        "Mistral 7B (Quick Analysis)": "mistralai/mistral-7b-instruct:free",
        "DeepSeek R1 (Reasoning)": "deepseek/deepseek-r1:free"
    }

    @staticmethod
    @st.cache_resource
    def get_llm(api_key: str, model_name: str):
        if not api_key:
            return None
        return OpenAI(
            api_token=api_key,
            api_base="https://openrouter.ai/api/v1",
            model=OpenRouterLLM.MODELS[model_name],
            temperature=0.1
        )

# --- 3. UTILITY FUNCTIONS ---

@st.cache_data
def load_data(file):
    """Robust data loader for CSV and Excel."""
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        else:
            return pd.read_excel(file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def generate_health_report(df):
    """Generates a quick data audit report."""
    report = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_cells": df.isnull().sum().sum(),
        "duplicate_rows": df.duplicated().sum(),
        "numeric_cols": list(df.select_dtypes(include=['number']).columns),
        "categorical_cols": list(df.select_dtypes(include=['object', 'category']).columns)
    }
    return report

def download_history():
    """Converts chat history to Markdown for download."""
    history = "# DataNexus AI Chat History\n\n"
    for msg in st.session_state.messages:
        role = "User" if msg["role"] == "user" else "AI"
        history += f"**{role}:** {msg['content']}\n\n"
    return history

# --- 4. STREAMLIT UI ---

def main():
    # Sidebar
    with st.sidebar:
        st.title("🧬 DataNexus AI")
        st.caption("v2.0 | Expert Edition")
        
        # Navigation
        app_mode = st.radio("Navigation", ["Dashboard", "AI Analyst", "Visual Explorer", "Data Audit"])
        
        st.markdown("---")
        
        # Config
        api_key = st.text_input("OpenRouter API Key", type="password", help="Required for AI features")
        selected_model = st.selectbox("AI Model", list(OpenRouterLLM.MODELS.keys()))
        
        uploaded_file = st.file_uploader("Upload Data (CSV/Excel)", type=['csv', 'xlsx'])
        
        # Advanced Settings
        with st.expander("⚙️ Engine Settings"):
            enforce_privacy = st.toggle("Enforce Privacy", value=True)
            enable_cache = st.toggle("Enable Caching", value=True)

    # Main App Logic
    if uploaded_file:
        df = load_data(uploaded_file)
        
        if df is not None:
            # --- VIEW: DASHBOARD (Default) ---
            if app_mode == "Dashboard":
                st.header("🚀 Executive Dashboard")
                
                # Metrics Row
                report = generate_health_report(df)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Rows", report['rows'])
                c2.metric("Columns", report['columns'])
                c3.metric("Missing Values", report['missing_cells'], delta_color="inverse")
                c4.metric("Duplicates", report['duplicate_rows'], delta_color="inverse")
                
                # Auto-Viz
                st.subheader("📈 Quick Insights")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.caption("Correlation Matrix (Numeric)")
                    if report['numeric_cols']:
                        fig, ax = plt.subplots()
                        sns.heatmap(df[report['numeric_cols']].corr(), annot=True, cmap='coolwarm', ax=ax)
                        st.pyplot(fig)
                    else:
                        st.info("No numeric columns for correlation.")

                with col2:
                    st.caption("Data Distribution (First Numeric Column)")
                    if report['numeric_cols']:
                        st.bar_chart(df[report['numeric_cols'][0]])
                    else:
                        st.info("No numeric columns to plot.")

            # --- VIEW: DATA AUDIT ---
            elif app_mode == "Data Audit":
                st.header("🛡️ Data Health Audit")
                
                st.subheader("Missing Value Analysis")
                missing = df.isnull().sum()
                st.bar_chart(missing[missing > 0])
                
                st.subheader("Data Types")
                st.json(df.dtypes.astype(str).to_dict())
                
                st.subheader("Head of Data")
                st.dataframe(df.head(), use_container_width=True)

            # --- VIEW: VISUAL EXPLORER (PyGWalker) ---
            elif app_mode == "Visual Explorer":
                st.header("🎨 Visual Explorer (Drag & Drop)")
                st.info("Use the interface below to build Tableau-style charts manually.")
                
                # PyGWalker Renderer
                if pyg:
                    pyg_html = pyg.to_html(df)
                    st.components.v1.html(pyg_html, height=800, scrolling=True)
                else:
                    st.warning("PyGWalker is not installed. Visual Explorer is disabled.")

            # --- VIEW: AI ANALYST (Chat) ---
            elif app_mode == "AI Analyst":
                st.header("🧠 Neural Analyst")
                
                if not api_key:
                    st.warning("Please enter your OpenRouter API Key in the sidebar.")
                    st.stop()
                
                # Initialize LLM
                llm = OpenRouterLLM.get_llm(api_key, selected_model)
                
                # SmartDataframe Init
                sdf = SmartDataframe(
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

                # Chat UI
                if "messages" not in st.session_state:
                    st.session_state.messages = []

                # Display Chat
                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]):
                        if msg.get("type") == "image":
                            st.image(msg["content"])
                        else:
                            st.markdown(msg["content"])

                # Input
                if prompt := st.chat_input("Ask: 'Plot sales over time' or 'Who is the top performer?'"):
                    # Add User Message
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    # Generate AI Response
                    with st.chat_message("assistant"):
                        with st.spinner("Analyzing..."):
                            try:
                                response = sdf.chat(prompt)
                                
                                # Image Response
                                if isinstance(response, str) and (response.endswith('.png') or response.endswith('.jpg')):
                                    st.image(response)
                                    st.session_state.messages.append({"role": "assistant", "content": response, "type": "image"})
                                
                                # DataFrame Response
                                elif isinstance(response, pd.DataFrame):
                                    st.dataframe(response)
                                    st.session_state.messages.append({"role": "assistant", "content": response.to_markdown(), "type": "text"})
                                
                                # Text Response
                                else:
                                    st.write(response)
                                    st.session_state.messages.append({"role": "assistant", "content": str(response), "type": "text"})
                            
                            except Exception as e:
                                st.error(f"Analysis Error: {e}")

                # History Export
                if st.session_state.messages:
                    st.download_button(
                        "Download Chat History",
                        data=download_history(),
                        file_name="datanexus_chat_history.md",
                        mime="text/markdown"
                    )

    else:
        # LANDING PAGE (No File)
        st.title("Welcome to DataNexus AI")
        st.markdown("""
        ### The Ultimate Free Analytics Platform
        Upload a CSV or Excel file to unlock:
        - **🧠 AI Chat:** Talk to your data using Gemini 2.0 or Llama 3.
        - **🎨 Visual Explorer:** Drag-and-drop Tableau-like builder.
        - **🛡️ Data Audit:** Auto-detect quality issues.
        - **🔒 Privacy First:** Your data stays local (if configured).
        """)
        
        # Sample Data Button
        if st.button("Load Demo Dataset"):
            # Create dummy data for demo
            dummy_data = pd.DataFrame({
                'Date': pd.date_range(start='1/1/2024', periods=100),
                'Sales': [x * 10 + (x % 5) * 50 for x in range(100)],
                'Category': ['Tech', 'Home', 'Office'] * 33 + ['Tech'],
                'Profit': [x * 2 for x in range(100)]
            })
            # Save to buffer to simulate upload (in a real app, we'd just load it)
            # For this demo, we can't easily set uploaded_file, so we just show a message
            st.success("For this demo, please upload a real CSV file to test the full engine!")

if __name__ == "__main__":
    main()