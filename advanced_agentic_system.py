"""
Advanced Agentic Data Analysis System
Based on the LangGraph multi-agent architecture shown in the video
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Annotated, Sequence, TypedDict
import json
import io
from datetime import datetime

# LangGraph and LangChain imports
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# Page configuration
st.set_page_config(
    page_title="🤖 Advanced Agentic Data Analyst",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# State definition for the multi-agent system
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    dataframe_json: str
    current_task: str
    analysis_results: Dict[str, Any]
    visualizations: List[Dict[str, Any]]
    insights: List[str]
    report: str
    next_agent: str

# Specialized Tools for Data Analysis
@tool
def analyze_data_structure(df_json: str) -> str:
    """Comprehensive data structure analysis tool"""
    try:
        df = pd.read_json(io.StringIO(df_json))
        
        analysis = {
            "basic_info": {
                "shape": list(df.shape),
                "columns": list(df.columns),
                "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2)
            },
            "column_analysis": {},
            "data_quality": {
                "missing_values": df.isnull().sum().to_dict(),
                "duplicate_rows": int(df.duplicated().sum()),
                "missing_percentage": round((df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 2)
            }
        }
        
        # Analyze each column
        for col in df.columns:
            col_info = {
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isnull().sum()),
                "null_percentage": round((df[col].isnull().sum() / len(df)) * 100, 2),
                "unique_count": int(df[col].nunique())
            }
            
            if df[col].dtype in ['int64', 'float64']:
                col_info.update({
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "mean": float(df[col].mean()),
                    "median": float(df[col].median()),
                    "std": float(df[col].std())
                })
            
            analysis["column_analysis"][col] = col_info
        
        return json.dumps(analysis, indent=2)
    except Exception as e:
        return f"Error in data structure analysis: {str(e)}"

@tool
def perform_statistical_analysis(df_json: str) -> str:
    """Advanced statistical analysis tool"""
    try:
        df = pd.read_json(io.StringIO(df_json))
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) == 0:
            return "No numeric columns found for statistical analysis"
        
        results = {
            "descriptive_stats": numeric_df.describe().to_dict(),
            "correlation_analysis": {},
            "distribution_analysis": {}
        }
        
        # Correlation analysis
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr()
            results["correlation_analysis"] = {
                "correlation_matrix": corr_matrix.to_dict(),
                "strong_correlations": []
            }
            
            # Find strong correlations
            for i, col1 in enumerate(corr_matrix.columns):
                for j, col2 in enumerate(corr_matrix.columns):
                    if i < j:
                        corr_val = corr_matrix.loc[col1, col2]
                        if abs(corr_val) > 0.7:
                            results["correlation_analysis"]["strong_correlations"].append({
                                "variable1": col1,
                                "variable2": col2,
                                "correlation": round(float(corr_val), 3)
                            })
        
        # Distribution analysis
        for col in numeric_df.columns:
            skew = float(numeric_df[col].skew())
            kurtosis = float(numeric_df[col].kurtosis())
            
            results["distribution_analysis"][col] = {
                "skewness": round(skew, 3),
                "kurtosis": round(kurtosis, 3),
                "distribution_type": "normal" if abs(skew) < 0.5 else ("right_skewed" if skew > 0 else "left_skewed")
            }
        
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Error in statistical analysis: {str(e)}"

@tool
def create_visualization_data(df_json: str, viz_type: str, columns: str) -> str:
    """Create visualization data for different chart types"""
    try:
        df = pd.read_json(io.StringIO(df_json))
        column_list = [col.strip() for col in columns.split(',') if col.strip() in df.columns]
        
        viz_data = {}
        
        if viz_type == "correlation_heatmap":
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 1:
                corr_matrix = numeric_df.corr()
                viz_data = {
                    "type": "correlation_heatmap",
                    "data": corr_matrix.to_dict(),
                    "title": "Correlation Heatmap"
                }
        
        elif viz_type == "distribution" and column_list:
            col = column_list[0]
            if df[col].dtype in ['int64', 'float64']:
                viz_data = {
                    "type": "histogram",
                    "data": df[col].values.tolist(),
                    "column": col,
                    "title": f"Distribution of {col}"
                }
        
        elif viz_type == "scatter" and len(column_list) >= 2:
            viz_data = {
                "type": "scatter",
                "x_data": df[column_list[0]].values.tolist(),
                "y_data": df[column_list[1]].values.tolist(),
                "x_column": column_list[0],
                "y_column": column_list[1],
                "title": f"{column_list[0]} vs {column_list[1]}"
            }
        
        return json.dumps(viz_data, indent=2)
    except Exception as e:
        return f"Error creating visualization data: {str(e)}"

@tool
def detect_outliers(df_json: str, column: str) -> str:
    """Detect outliers in a specific column"""
    try:
        df = pd.read_json(io.StringIO(df_json))
        
        if column not in df.columns or df[column].dtype not in ['int64', 'float64']:
            return f"Column {column} not found or not numeric"
        
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        
        result = {
            "column": column,
            "total_outliers": len(outliers),
            "outlier_percentage": round((len(outliers) / len(df)) * 100, 2),
            "outlier_bounds": {
                "lower": float(lower_bound),
                "upper": float(upper_bound)
            },
            "sample_outliers": outliers[column].head(10).tolist()
        }
        
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error detecting outliers: {str(e)}"

# Specialized Agents
class DataAnalysisAgent:
    def __init__(self, llm):
        self.llm = llm
        self.tools = [analyze_data_structure, perform_statistical_analysis, detect_outliers]
        self.llm_with_tools = llm.bind_tools(self.tools)
        
    def analyze(self, state: AgentState) -> Dict[str, Any]:
        system_prompt = """You are a Data Analysis Agent. Your job is to thoroughly analyze the dataset structure and perform statistical analysis.

        Available tools:
        1. analyze_data_structure - Get comprehensive info about the dataset
        2. perform_statistical_analysis - Calculate statistics and correlations  
        3. detect_outliers - Find outliers in numeric columns

        Always start with data structure analysis, then perform statistical analysis. Look for interesting patterns and anomalies.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Analyze this dataset. Current task: {state.get('current_task', 'general analysis')}")
        ]
        
        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response], "next_agent": "visualization"}

class VisualizationAgent:
    def __init__(self, llm):
        self.llm = llm
        self.tools = [create_visualization_data]
        self.llm_with_tools = llm.bind_tools(self.tools)
        
    def create_visualizations(self, state: AgentState) -> Dict[str, Any]:
        system_prompt = """You are a Visualization Agent. Create appropriate visualizations based on the data analysis results.

        Available tools:
        1. create_visualization_data - Generate data for different chart types

        Create multiple visualizations:
        - Correlation heatmap for numeric variables
        - Distribution plots for key variables
        - Scatter plots for relationships
        
        Base your decisions on the analysis results from the previous agent.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Create comprehensive visualizations for this dataset based on the analysis results.")
        ]
        
        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response], "next_agent": "insights"}

class InsightAgent:
    def __init__(self, llm):
        self.llm = llm
        
    def generate_insights(self, state: AgentState) -> Dict[str, Any]:
        system_prompt = """You are an Insight Generation Agent. Analyze all the results and generate key insights.

        Based on the data analysis and visualizations, provide:
        1. Key findings and patterns
        2. Anomalies and outliers
        3. Relationships between variables
        4. Data quality observations
        5. Business implications (if applicable)

        Generate clear, actionable insights.
        """
        
        # Summarize previous analysis
        analysis_summary = "Previous agents have analyzed the data structure and created visualizations."
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Generate insights based on: {analysis_summary}")
        ]
        
        response = self.llm.invoke(messages)
        return {"messages": [response], "next_agent": "report"}

class ReportAgent:
    def __init__(self, llm):
        self.llm = llm
        
    def generate_report(self, state: AgentState) -> Dict[str, Any]:
        system_prompt = """You are a Report Generation Agent. Create a comprehensive data analysis report.

        Structure your report with:
        1. Executive Summary
        2. Dataset Overview
        3. Key Findings
        4. Statistical Analysis Results
        5. Visualizations Summary
        6. Insights and Recommendations
        7. Conclusion

        Make it professional and comprehensive.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Generate a comprehensive data analysis report based on all previous agent results.")
        ]
        
        response = self.llm.invoke(messages)
        return {"messages": [response], "report": response.content, "next_agent": "end"}

# Supervisor Agent
class SupervisorAgent:
    def __init__(self, llm):
        self.llm = llm
        
    def route_task(self, state: AgentState) -> str:
        """Route to the next appropriate agent"""
        next_agent = state.get("next_agent", "analysis")
        
        if next_agent == "end":
            return "end"
        elif next_agent == "visualization":
            return "visualization_agent"
        elif next_agent == "insights":
            return "insight_agent"
        elif next_agent == "report":
            return "report_agent"
        else:
            return "analysis_agent"

# Create the Multi-Agent Workflow
def create_agentic_workflow(api_key: str):
    """Create the multi-agent LangGraph workflow"""
    
    # Initialize LLM
    llm = ChatAnthropic(
        model="claude-3-haiku-20240307",
        temperature=0.1,
        api_key=api_key,
        max_tokens=2000
    )
    
    # Initialize agents
    analysis_agent = DataAnalysisAgent(llm)
    viz_agent = VisualizationAgent(llm)
    insight_agent = InsightAgent(llm)
    report_agent = ReportAgent(llm)
    supervisor = SupervisorAgent(llm)
    
    # Create workflow
    workflow = StateGraph(AgentState)
    
    # Add agent nodes
    workflow.add_node("analysis_agent", analysis_agent.analyze)
    workflow.add_node("visualization_agent", viz_agent.create_visualizations)
    workflow.add_node("insight_agent", insight_agent.generate_insights)
    workflow.add_node("report_agent", report_agent.generate_report)
    workflow.add_node("tools", ToolNode([
        analyze_data_structure, 
        perform_statistical_analysis, 
        create_visualization_data, 
        detect_outliers
    ]))
    
    # Define routing logic
    def should_use_tools(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        return supervisor.route_task(state)
    
    # Set entry point
    workflow.set_entry_point("analysis_agent")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "analysis_agent",
        should_use_tools,
        {
            "tools": "tools",
            "visualization_agent": "visualization_agent",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "visualization_agent", 
        should_use_tools,
        {
            "tools": "tools",
            "insight_agent": "insight_agent",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "insight_agent",
        should_use_tools,
        {
            "report_agent": "report_agent",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "report_agent",
        should_use_tools,
        {
            "end": END
        }
    )
    
    # Tools always route back to supervisor
    workflow.add_conditional_edges(
        "tools",
        should_use_tools,
        {
            "analysis_agent": "analysis_agent",
            "visualization_agent": "visualization_agent", 
            "insight_agent": "insight_agent",
            "report_agent": "report_agent",
            "end": END
        }
    )
    
    # Compile with memory
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

# Streamlit App
def main():
    st.title("🧠 Advanced Agentic Data Analysis System")
    st.markdown("**Multi-Agent AI System for Comprehensive Data Analysis**")
    
    # Initialize session state
    if 'workflow' not in st.session_state:
        st.session_state.workflow = None
    if 'analysis_running' not in st.session_state:
        st.session_state.analysis_running = False
    if 'results' not in st.session_state:
        st.session_state.results = None
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            help="Required for Claude agents"
        )
        
        if api_key:
            if st.session_state.workflow is None:
                with st.spinner("Initializing Multi-Agent System..."):
                    st.session_state.workflow = create_agentic_workflow(api_key)
                st.success("✅ Multi-Agent System Ready!")
        
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
        
        if uploaded_file and st.session_state.workflow:
            df = pd.read_csv(uploaded_file)
            st.success(f"📊 Data loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
            
            if st.button("🚀 Start Agentic Analysis", type="primary"):
                st.session_state.analysis_running = True
                
                # Run the multi-agent workflow
                with st.spinner("🤖 Agents are analyzing your data..."):
                    try:
                        config = {"configurable": {"thread_id": "analysis_session"}}
                        initial_state = {
                            "messages": [HumanMessage(content="Perform comprehensive data analysis")],
                            "dataframe_json": df.to_json(),
                            "current_task": "comprehensive_analysis",
                            "analysis_results": {},
                            "visualizations": [],
                            "insights": [],
                            "report": "",
                            "next_agent": "analysis"
                        }
                        
                        result = st.session_state.workflow.invoke(initial_state, config)
                        st.session_state.results = result
                        st.session_state.analysis_running = False
                        
                    except Exception as e:
                        st.error(f"Error in analysis: {str(e)}")
                        st.session_state.analysis_running = False
    
    # Main content
    if st.session_state.analysis_running:
        st.info("🤖 Multi-Agent Analysis in Progress...")
        st.markdown("**Agents Working:**")
        st.markdown("- 📊 Data Analysis Agent: Analyzing structure and statistics")
        st.markdown("- 📈 Visualization Agent: Creating charts and graphs") 
        st.markdown("- 💡 Insight Agent: Generating key findings")
        st.markdown("- 📝 Report Agent: Compiling comprehensive report")
    
    elif st.session_state.results:
        st.success("✅ Multi-Agent Analysis Complete!")
        
        # Display results
        st.header("📋 Agent Communications")
        for i, message in enumerate(st.session_state.results.get("messages", [])):
            if isinstance(message, AIMessage):
                with st.expander(f"Agent Response {i+1}"):
                    st.write(message.content)
        
        # Display final report if available
        if st.session_state.results.get("report"):
            st.header("📄 Final Analysis Report")
            st.markdown(st.session_state.results["report"])
    
    else:
        st.info("🔧 Configure your API key and upload data to start the multi-agent analysis!")
        
        st.header("🎯 What This System Does")
        st.markdown("""
        This advanced agentic system uses **multiple specialized AI agents**:
        
        1. **📊 Data Analysis Agent** - Analyzes structure, statistics, and quality
        2. **📈 Visualization Agent** - Creates appropriate charts and graphs  
        3. **💡 Insight Agent** - Generates key findings and patterns
        4. **📝 Report Agent** - Compiles comprehensive analysis report
        5. **🎯 Supervisor Agent** - Coordinates the entire workflow
        
        Each agent specializes in specific tasks and communicates with others to provide comprehensive analysis.
        """)

if __name__ == "__main__":
    main()