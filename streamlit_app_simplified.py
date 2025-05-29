"""
Agentic Data Analysis - Streamlit Application
Updated version using Claude/Anthropic instead of OpenAI
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import io
from typing import Dict, Any, List

from langchain_core.messages import HumanMessage, AIMessage
from agent_nodes import DataAnalysisAgent, get_suggested_analyses, format_analysis_results

# Page configuration
st.set_page_config(
    page_title="🤖 Agentic Data Analyst (Claude)",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        background-color: #f8f9fa;
    }
    
    .analysis-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
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
def initialize_session_state():
    """Initialize all necessary session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'dataframe' not in st.session_state:
        st.session_state.dataframe = None
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'workflow' not in st.session_state:
        st.session_state.workflow = None
    if 'current_plots' not in st.session_state:
        st.session_state.current_plots = []
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []

def display_header():
    """Display the main application header"""
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Agentic Data Analyst (Claude)</h1>
        <p>Upload your data and let Claude perform intelligent analysis with advanced visualizations</p>
    </div>
    """, unsafe_allow_html=True)

def display_sidebar():
    """Display the sidebar with file upload and configuration"""
    with st.sidebar:
        st.header("📁 Data Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file to start your data analysis journey"
        )
        
        if uploaded_file is not None:
            try:
                # Load the dataset
                df = pd.read_csv(uploaded_file)
                st.session_state.dataframe = df
                
                # Display dataset info
                st.success(f"✅ Dataset loaded successfully!")
                
                # Dataset metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Rows", f"{df.shape[0]:,}")
                with col2:
                    st.metric("Columns", f"{df.shape[1]}")
                
                # Quick data preview
                with st.expander("👀 Data Preview"):
                    st.dataframe(df.head(), use_container_width=True)
                
                # Column information
                with st.expander("📋 Column Info"):
                    col_info = pd.DataFrame({
                        'Column': df.columns,
                        'Type': df.dtypes.astype(str),
                        'Non-Null': df.count(),
                        'Null %': ((df.isnull().sum() / len(df)) * 100).round(2)
                    })
                    st.dataframe(col_info, use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
                st.session_state.dataframe = None
        
        st.header("🔧 Configuration")
        
        # API Key input - Updated for Anthropic
        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            help="Enter your Anthropic API key to enable Claude",
            placeholder="sk-ant-..."
        )
        
        if api_key:
            # Initialize the agent
            try:
                if st.session_state.agent is None:
                    st.session_state.agent = DataAnalysisAgent(api_key)
                    st.session_state.workflow = st.session_state.agent.create_workflow()
                st.success("✅ Claude Agent Ready!")
            except Exception as e:
                st.error(f"❌ Error initializing Claude agent: {str(e)}")
        
        # Model selection - Updated for Claude models
        model_choice = st.selectbox(
            "Claude Model",
            [
                "claude-3-haiku-20240307",
                "claude-3-sonnet-20240229", 
                "claude-3-opus-20240229"
            ],
            help="Choose the Claude model for analysis"
        )
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            temperature = st.slider("Response Creativity", 0.0, 1.0, 0.1)
            max_tokens = st.number_input("Max Response Length", 1000, 8000, 4000)

def display_quick_insights():
    """Display quick insights about the loaded dataset"""
    if st.session_state.dataframe is not None:
        df = st.session_state.dataframe
        
        st.subheader("📊 Quick Insights")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>{:,}</h3>
                <p>Total Records</p>
            </div>
            """.format(len(df)), unsafe_allow_html=True)
        
        with col2:
            numeric_cols = len(df.select_dtypes(include=['number']).columns)
            st.markdown("""
            <div class="metric-card">
                <h3>{}</h3>
                <p>Numeric Columns</p>
            </div>
            """.format(numeric_cols), unsafe_allow_html=True)
        
        with col3:
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
            st.markdown("""
            <div class="metric-card">
                <h3>{:.1f}%</h3>
                <p>Missing Data</p>
            </div>
            """.format(missing_pct), unsafe_allow_html=True)
        
        with col4:
            duplicates = df.duplicated().sum()
            st.markdown("""
            <div class="metric-card">
                <h3>{}</h3>
                <p>Duplicates</p>
            </div>
            """.format(duplicates), unsafe_allow_html=True)

def display_suggested_analyses():
    """Display suggested analysis prompts based on the dataset"""
    if st.session_state.dataframe is not None:
        st.subheader("💡 Suggested Analyses")
        
        suggestions = get_suggested_analyses(st.session_state.dataframe)
        
        cols = st.columns(3)
        for i, suggestion in enumerate(suggestions[:6]):  # Show top 6 suggestions
            with cols[i % 3]:
                if st.button(suggestion["title"], key=f"suggestion_{i}", use_container_width=True):
                    # Add the suggested prompt to chat
                    st.session_state.messages.append({
                        "role": "user", 
                        "content": suggestion["prompt"]
                    })
                    st.rerun()

def process_agent_response(user_input: str):
    """Process user input through the agent and return response"""
    if not st.session_state.agent or not st.session_state.workflow:
        return "Please configure your Anthropic API key first."
    
    try:
        # Prepare the state
        df_json = st.session_state.dataframe.to_json() if st.session_state.dataframe is not None else ""
        
        config = {"configurable": {"thread_id": "main_conversation"}}
        state = {
            "messages": [HumanMessage(content=user_input)],
            "dataframe_json": df_json
        }
        
        # Invoke the workflow
        result = st.session_state.workflow.invoke(state, config)
        
        # Extract AI response
        ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
        if ai_messages:
            return ai_messages[-1].content
        else:
            return "I encountered an issue processing your request. Please try again."
            
    except Exception as e:
        return f"Error processing request: {str(e)}"

def display_chat_interface():
    """Display the main chat interface"""
    st.subheader("💬 Chat with Your Data (Claude)")
    
    # Check if requirements are met
    if st.session_state.dataframe is None:
        st.info("👆 Please upload a CSV file to start analyzing your data!")
        return
    
    if st.session_state.agent is None:
        st.warning("🔑 Please enter your Anthropic API key in the sidebar to enable Claude!")
        return
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.write(message["content"])
                else:
                    # Display assistant response with better formatting
                    st.markdown(message["content"])
                    
                    # Check if there are any plots to display
                    if "visualization" in message["content"].lower() and "successfully" in message["content"].lower():
                        # This is a placeholder - in a real implementation, you'd extract and display the actual plot
                        st.info("📊 Visualization would be displayed here")
    
    # Chat input
    if prompt := st.chat_input("Ask Claude anything about your data..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Claude is analyzing your data..."):
                response = process_agent_response(prompt)
                st.markdown(response)
                
                # Add assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": response})

def display_analysis_history():
    """Display history of analyses performed"""
    if st.session_state.analysis_history:
        st.subheader("📈 Analysis History")
        
        for i, analysis in enumerate(reversed(st.session_state.analysis_history[-5:])):  # Show last 5
            with st.expander(f"Analysis {len(st.session_state.analysis_history) - i}: {analysis.get('title', 'Untitled')}"):
                st.write(f"**Query:** {analysis.get('query', 'N/A')}")
                st.write(f"**Result:** {analysis.get('result', 'N/A')}")
                st.write(f"**Timestamp:** {analysis.get('timestamp', 'N/A')}")

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Display header
    display_header()
    
    # Display sidebar
    display_sidebar()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat interface
        display_chat_interface()
        
        # Analysis history
        display_analysis_history()
    
    with col2:
        # Quick insights
        display_quick_insights()
        
        # Suggested analyses
        display_suggested_analyses()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**🚀 Built with LangGraph & Streamlit** | "
        "💡 Powered by Claude (Anthropic) | "
        "📊 Advanced Data Analysis Made Simple"
    )

if __name__ == "__main__":
    main()