"""
Working Agentic Data Analysis - Just like the video!
This version actually performs data science tasks and shows visualizations
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import json
import io
from typing import Dict, Any, List
import seaborn as sns
import matplotlib.pyplot as plt

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate

# Page configuration
st.set_page_config(
    page_title="🤖 Working Data Scientist AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'dataframe' not in st.session_state:
    st.session_state.dataframe = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

# Data Science Functions
def analyze_data_structure(df):
    """Analyze the structure of the dataset"""
    analysis = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": df.duplicated().sum(),
        "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
    }
    
    # Categorize columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    analysis.update({
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "datetime_columns": datetime_cols
    })
    
    return analysis

def perform_statistical_analysis(df):
    """Perform comprehensive statistical analysis"""
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) == 0:
        return "No numeric columns found for statistical analysis."
    
    stats = {
        "descriptive_stats": numeric_df.describe().to_dict(),
        "correlations": numeric_df.corr().to_dict(),
        "skewness": numeric_df.skew().to_dict(),
        "kurtosis": numeric_df.kurtosis().to_dict()
    }
    
    return stats

def create_visualizations(df):
    """Create multiple visualizations automatically"""
    plots = []
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 1. Correlation Heatmap (if multiple numeric columns)
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        fig_corr = px.imshow(
            corr_matrix, 
            title="🔥 Correlation Heatmap",
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        plots.append(("Correlation Heatmap", fig_corr))
    
    # 2. Distribution plots for numeric columns
    for col in numeric_cols[:3]:  # Show first 3 numeric columns
        fig_hist = px.histogram(
            df, 
            x=col, 
            title=f"📊 Distribution of {col}",
            marginal="box"
        )
        plots.append((f"Distribution of {col}", fig_hist))
    
    # 3. Bar plots for categorical columns
    for col in categorical_cols[:2]:  # Show first 2 categorical columns
        value_counts = df[col].value_counts().head(10)
        fig_bar = px.bar(
            x=value_counts.index,
            y=value_counts.values,
            title=f"📈 Top Values in {col}",
            labels={'x': col, 'y': 'Count'}
        )
        plots.append((f"Top Values in {col}", fig_bar))
    
    # 4. Scatter plot (if we have at least 2 numeric columns)
    if len(numeric_cols) >= 2:
        fig_scatter = px.scatter(
            df,
            x=numeric_cols[0],
            y=numeric_cols[1],
            title=f"🎯 {numeric_cols[0]} vs {numeric_cols[1]}",
            color=categorical_cols[0] if categorical_cols else None
        )
        plots.append((f"{numeric_cols[0]} vs {numeric_cols[1]}", fig_scatter))
    
    return plots

def generate_insights(df, stats):
    """Generate data science insights"""
    insights = []
    
    # Dataset overview
    insights.append(f"📋 **Dataset Overview**: {df.shape[0]:,} rows and {df.shape[1]} columns")
    
    # Missing data insights
    missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    if missing_pct > 5:
        insights.append(f"⚠️ **Data Quality Alert**: {missing_pct:.1f}% of data is missing")
    else:
        insights.append(f"✅ **Data Quality**: Only {missing_pct:.1f}% missing data")
    
    # Numeric insights
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        insights.append(f"🔢 **Numeric Analysis**: Found {len(numeric_cols)} numeric variables")
        
        # Find highly correlated pairs
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
            
            if high_corr_pairs:
                insights.append(f"🔗 **Strong Correlations Found**: {len(high_corr_pairs)} pairs with |correlation| > 0.7")
    
    # Categorical insights
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        insights.append(f"📊 **Categorical Analysis**: Found {len(categorical_cols)} categorical variables")
    
    # Outlier detection
    for col in numeric_cols[:3]:  # Check first 3 numeric columns
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
        if len(outliers) > 0:
            outlier_pct = (len(outliers) / len(df)) * 100
            insights.append(f"🚨 **Outliers in {col}**: {len(outliers)} ({outlier_pct:.1f}%) potential outliers detected")
    
    return insights

# Main App
def main():
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Working Data Scientist AI</h1>
        <p>Upload your data and watch me perform real data science analysis!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Data Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file to start automatic analysis"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.dataframe = df
                st.success(f"✅ Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
                
                # Quick preview
                with st.expander("👀 Data Preview"):
                    st.dataframe(df.head())
                    
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
        
        st.header("🔧 Configuration")
        
        # Claude API Key
        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            help="Enter your Anthropic API key for Claude",
            placeholder="sk-ant-..."
        )
        
        if api_key:
            st.success("✅ Claude Ready!")
        
        # Auto-analysis button
        if st.session_state.dataframe is not None:
            if st.button("🚀 Start Automatic Analysis", type="primary"):
                st.session_state.analysis_complete = False
                st.rerun()
    
    # Main content
    if st.session_state.dataframe is not None:
        df = st.session_state.dataframe
        
        # Automatic analysis
        if not st.session_state.analysis_complete:
            st.header("🔍 Performing Automatic Data Science Analysis...")
            
            with st.spinner("Analyzing your data..."):
                # 1. Structure Analysis
                st.subheader("📋 Dataset Structure Analysis")
                structure = analyze_data_structure(df)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Rows", f"{structure['shape'][0]:,}")
                with col2:
                    st.metric("Total Columns", structure['shape'][1])
                with col3:
                    st.metric("Numeric Columns", len(structure['numeric_columns']))
                with col4:
                    st.metric("Categorical Columns", len(structure['categorical_columns']))
                
                # 2. Statistical Analysis
                st.subheader("📊 Statistical Analysis")
                stats = perform_statistical_analysis(df)
                
                if isinstance(stats, dict):
                    # Show correlation matrix
                    if len(df.select_dtypes(include=[np.number]).columns) > 1:
                        st.write("**🔗 Correlation Matrix:**")
                        corr_df = pd.DataFrame(stats['correlations'])
                        st.dataframe(corr_df.style.background_gradient(cmap='RdBu'))
                else:
                    st.write(stats)
                
                # 3. Generate Visualizations
                st.subheader("📈 Automatic Visualizations")
                plots = create_visualizations(df)
                
                # Display all plots
                for plot_name, fig in plots:
                    st.plotly_chart(fig, use_container_width=True)
                
                # 4. Generate Insights
                st.subheader("💡 Key Insights")
                insights = generate_insights(df, stats)
                
                for insight in insights:
                    st.write(insight)
                
                # 5. Recommendations
                st.subheader("🎯 Recommendations")
                recommendations = [
                    "🔍 **Further Analysis**: Consider exploring relationships between variables with highest correlations",
                    "📊 **Data Quality**: Address missing values if percentage is high",
                    "🚨 **Outliers**: Investigate outliers - they might reveal interesting patterns or data quality issues",
                    "📈 **Feature Engineering**: Consider creating new features from existing ones",
                    "🤖 **Machine Learning**: This dataset could be suitable for predictive modeling"
                ]
                
                for rec in recommendations:
                    st.write(rec)
                
                st.session_state.analysis_complete = True
                st.balloons()
        
        # Chat interface
        st.header("💬 Ask Me Anything About Your Data")
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("What would you like to know about your data?"):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate response based on data
            with st.chat_message("assistant"):
                if api_key:
                    try:
                        # Use Claude for intelligent responses
                        llm = ChatAnthropic(
                            model="claude-3-haiku-20240307",
                            temperature=0.1,
                            api_key=api_key,
                            max_tokens=1000
                        )
                        
                        # Create context about the data
                        data_context = f"""
                        Dataset Info:
                        - Shape: {df.shape}
                        - Columns: {list(df.columns)}
                        - Numeric columns: {df.select_dtypes(include=[np.number]).columns.tolist()}
                        - Categorical columns: {df.select_dtypes(include=['object']).columns.tolist()}
                        - Missing values: {df.isnull().sum().sum()}
                        """
                        
                        system_prompt = f"""You are a data scientist analyzing this dataset:
                        {data_context}
                        
                        Answer the user's question about this data with specific insights and suggestions.
                        Be concise but informative."""
                        
                        response = llm.invoke([
                            HumanMessage(content=f"{system_prompt}\n\nUser question: {prompt}")
                        ])
                        
                        st.write(response.content)
                        st.session_state.messages.append({"role": "assistant", "content": response.content})
                        
                    except Exception as e:
                        response = f"I can see your data has {df.shape[0]} rows and {df.shape[1]} columns. To provide more detailed analysis, please ensure your Claude API key is configured correctly. Error: {str(e)}"
                        st.write(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    response = f"I can see your data has {df.shape[0]} rows and {df.shape[1]} columns. To provide more detailed analysis, please add your Claude API key in the sidebar."
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
    
    else:
        st.info("👆 Please upload a CSV file to start the automatic data science analysis!")
        
        # Show example
        st.subheader("🎬 What This App Does:")
        st.write("""
        1. **📊 Automatic Structure Analysis** - Analyzes your data types, missing values, etc.
        2. **📈 Statistical Analysis** - Calculates correlations, distributions, and summary statistics
        3. **🎨 Auto-Generated Visualizations** - Creates correlation heatmaps, histograms, scatter plots
        4. **💡 Intelligent Insights** - Identifies patterns, outliers, and data quality issues
        5. **🤖 Interactive Chat** - Ask questions about your data using natural language
        """)

if __name__ == "__main__":
    main()