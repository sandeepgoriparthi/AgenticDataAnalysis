"""
Agent Nodes for Data Analysis - Updated for Claude/Anthropic
This file contains the core agent logic separated from the Streamlit interface
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Annotated, Sequence, Optional
import io
import json
from dataclasses import dataclass

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import AnyMessage
from langgraph.checkpoint.memory import MemorySaver

@dataclass
class DataAnalysisState:
    """State management for the data analysis agent workflow"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    dataframe_json: str = ""
    current_analysis: Dict[str, Any] = None
    visualization_data: Dict[str, Any] = None
    analysis_history: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.current_analysis is None:
            self.current_analysis = {}
        if self.visualization_data is None:
            self.visualization_data = {}
        if self.analysis_history is None:
            self.analysis_history = []

class DataAnalysisTools:
    """Collection of tools for data analysis"""
    
    @staticmethod
    @tool
    def analyze_dataframe_structure(df_json: str) -> str:
        """Analyze the structure and basic information of a pandas DataFrame"""
        try:
            df = pd.read_json(io.StringIO(df_json))
            
            # Basic structure analysis
            structure_info = {
                "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
                "columns": list(df.columns),
                "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
                "null_counts": {col: int(count) for col, count in df.isnull().sum().items()},
                "duplicate_rows": int(df.duplicated().sum())
            }
            
            # Column categorization
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            
            column_types = {
                "numeric": numeric_cols,
                "categorical": categorical_cols,
                "datetime": datetime_cols
            }
            
            # Sample data
            sample_data = df.head(3).to_dict('records')
            
            analysis_result = {
                "structure": structure_info,
                "column_types": column_types,
                "sample_data": sample_data,
                "summary": f"Dataset has {structure_info['shape']['rows']} rows and {structure_info['shape']['columns']} columns. "
                          f"Contains {len(numeric_cols)} numeric, {len(categorical_cols)} categorical, "
                          f"and {len(datetime_cols)} datetime columns."
            }
            
            return json.dumps(analysis_result, indent=2)
            
        except Exception as e:
            return f"Error analyzing DataFrame structure: {str(e)}"
    
    @staticmethod
    @tool
    def perform_statistical_analysis(df_json: str, columns: str = "", analysis_type: str = "descriptive") -> str:
        """Perform comprehensive statistical analysis on specified columns or entire dataset"""
        try:
            df = pd.read_json(io.StringIO(df_json))
            
            if columns:
                column_list = [col.strip() for col in columns.split(',')]
                column_list = [col for col in column_list if col in df.columns]
            else:
                column_list = df.columns.tolist()
            
            results = {}
            
            for col in column_list:
                if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    # Numerical analysis
                    col_data = df[col].dropna()
                    if len(col_data) > 0:
                        results[col] = {
                            "type": "numerical",
                            "count": int(col_data.count()),
                            "mean": float(col_data.mean()),
                            "median": float(col_data.median()),
                            "std": float(col_data.std()),
                            "min": float(col_data.min()),
                            "max": float(col_data.max()),
                            "q25": float(col_data.quantile(0.25)),
                            "q75": float(col_data.quantile(0.75)),
                            "skewness": float(col_data.skew()),
                            "kurtosis": float(col_data.kurtosis())
                        }
                        
                        if analysis_type == "advanced":
                            # Add outlier detection
                            iqr = results[col]["q75"] - results[col]["q25"]
                            lower_bound = results[col]["q25"] - 1.5 * iqr
                            upper_bound = results[col]["q75"] + 1.5 * iqr
                            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                            results[col]["outliers_count"] = len(outliers)
                            results[col]["outliers_percentage"] = (len(outliers) / len(col_data)) * 100
                
                else:
                    # Categorical analysis
                    col_data = df[col].dropna()
                    if len(col_data) > 0:
                        value_counts = col_data.value_counts()
                        results[col] = {
                            "type": "categorical",
                            "count": int(col_data.count()),
                            "unique_values": int(col_data.nunique()),
                            "most_frequent": str(value_counts.index[0]) if len(value_counts) > 0 else "N/A",
                            "most_frequent_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                            "top_5_values": {str(k): int(v) for k, v in value_counts.head(5).items()}
                        }
            
            return json.dumps(results, indent=2)
            
        except Exception as e:
            return f"Error performing statistical analysis: {str(e)}"
    
    @staticmethod
    @tool
    def create_visualization(df_json: str, chart_type: str, x_column: str, y_column: str = "", 
                           color_column: str = "", title: str = "") -> str:
        """Create various types of visualizations from the DataFrame"""
        try:
            df = pd.read_json(io.StringIO(df_json))
            
            # Validate columns exist
            if x_column not in df.columns:
                return f"Error: Column '{x_column}' not found in DataFrame"
            
            if y_column and y_column not in df.columns:
                return f"Error: Column '{y_column}' not found in DataFrame"
            
            if color_column and color_column not in df.columns:
                return f"Error: Column '{color_column}' not found in DataFrame"
            
            # Generate title if not provided
            if not title:
                if chart_type.lower() == "histogram":
                    title = f"Distribution of {x_column}"
                elif y_column:
                    title = f"{chart_type.title()}: {x_column} vs {y_column}"
                else:
                    title = f"{chart_type.title()} of {x_column}"
            
            # Create visualization based on type
            fig = None
            
            if chart_type.lower() == "histogram":
                fig = px.histogram(df, x=x_column, title=title,
                                 color=color_column if color_column else None)
                
            elif chart_type.lower() == "scatter":
                if not y_column:
                    return "Error: Scatter plot requires both x and y columns"
                fig = px.scatter(df, x=x_column, y=y_column, 
                               color=color_column if color_column else None, title=title)
                
            elif chart_type.lower() == "bar":
                if y_column:
                    # Bar chart with specific y values
                    fig = px.bar(df, x=x_column, y=y_column, title=title,
                               color=color_column if color_column else None)
                else:
                    # Bar chart of value counts
                    value_counts = df[x_column].value_counts().head(20)
                    fig = px.bar(x=value_counts.index, y=value_counts.values,
                               title=f"Count of {x_column}", labels={'x': x_column, 'y': 'Count'})
                
            elif chart_type.lower() == "line":
                if not y_column:
                    return "Error: Line plot requires both x and y columns"
                fig = px.line(df, x=x_column, y=y_column, title=title,
                            color=color_column if color_column else None)
                
            elif chart_type.lower() == "box":
                if y_column:
                    fig = px.box(df, x=x_column, y=y_column, title=title)
                else:
                    fig = px.box(df, y=x_column, title=f"Box Plot of {x_column}")
                
            elif chart_type.lower() == "violin":
                if y_column:
                    fig = px.violin(df, x=x_column, y=y_column, title=title)
                else:
                    fig = px.violin(df, y=x_column, title=f"Violin Plot of {x_column}")
                
            elif chart_type.lower() == "correlation":
                # Correlation heatmap for numeric columns
                numeric_df = df.select_dtypes(include=[np.number])
                if len(numeric_df.columns) < 2:
                    return "Error: Correlation plot requires at least 2 numeric columns"
                corr_matrix = numeric_df.corr()
                fig = px.imshow(corr_matrix, title="Correlation Heatmap", 
                              color_continuous_scale='RdBu', aspect='auto')
                
            else:
                return f"Unsupported chart type: {chart_type}. Available types: histogram, scatter, bar, line, box, violin, correlation"
            
            if fig is None:
                return f"Failed to create {chart_type} visualization"
            
            return f"Successfully created {chart_type} visualization: '{title}'. " \
                   f"Chart shows relationship between {x_column}" + \
                   (f" and {y_column}" if y_column else "") + \
                   (f" colored by {color_column}" if color_column else "") + "."
            
        except Exception as e:
            return f"Error creating visualization: {str(e)}"
    
    @staticmethod
    @tool
    def filter_and_query_data(df_json: str, query: str, operation: str = "filter") -> str:
        """Filter and query the DataFrame based on conditions or perform aggregations"""
        try:
            df = pd.read_json(io.StringIO(df_json))
            
            if operation == "filter":
                # Handle different types of filtering conditions
                if " > " in query:
                    column, value = query.split(" > ")
                    column = column.strip()
                    value = float(value.strip())
                    filtered_df = df[df[column] > value]
                    condition = f"{column} > {value}"
                    
                elif " < " in query:
                    column, value = query.split(" < ")
                    column = column.strip()
                    value = float(value.strip())
                    filtered_df = df[df[column] < value]
                    condition = f"{column} < {value}"
                    
                elif " >= " in query:
                    column, value = query.split(" >= ")
                    column = column.strip()
                    value = float(value.strip())
                    filtered_df = df[df[column] >= value]
                    condition = f"{column} >= {value}"
                    
                elif " <= " in query:
                    column, value = query.split(" <= ")
                    column = column.strip()
                    value = float(value.strip())
                    filtered_df = df[df[column] <= value]
                    condition = f"{column} <= {value}"
                    
                elif " == " in query:
                    column, value = query.split(" == ")
                    column = column.strip()
                    value = value.strip().strip("'\"")
                    # Try to convert to appropriate type
                    try:
                        value = float(value)
                    except ValueError:
                        pass  # Keep as string
                    filtered_df = df[df[column] == value]
                    condition = f"{column} == {value}"
                    
                elif " != " in query:
                    column, value = query.split(" != ")
                    column = column.strip()
                    value = value.strip().strip("'\"")
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                    filtered_df = df[df[column] != value]
                    condition = f"{column} != {value}"
                    
                else:
                    return f"Query format not recognized. Use formats like 'column > value', 'column == value', etc."
                
                result = {
                    "operation": "filter",
                    "condition": condition,
                    "original_rows": len(df),
                    "filtered_rows": len(filtered_df),
                    "percentage_remaining": (len(filtered_df) / len(df)) * 100,
                    "sample_data": filtered_df.head(5).to_dict('records') if len(filtered_df) > 0 else []
                }
                
            elif operation == "groupby":
                # Handle group by operations
                if " by " in query.lower():
                    parts = query.lower().split(" by ")
                    agg_part = parts[0].strip()
                    group_col = parts[1].strip()
                    
                    if group_col not in df.columns:
                        return f"Column '{group_col}' not found for grouping"
                    
                    if "count" in agg_part:
                        grouped = df.groupby(group_col).size().reset_index(name='count')
                    elif "mean" in agg_part or "average" in agg_part:
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) == 0:
                            return "No numeric columns found for mean calculation"
                        grouped = df.groupby(group_col)[numeric_cols].mean().reset_index()
                    elif "sum" in agg_part:
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) == 0:
                            return "No numeric columns found for sum calculation"
                        grouped = df.groupby(group_col)[numeric_cols].sum().reset_index()
                    else:
                        grouped = df.groupby(group_col).size().reset_index(name='count')
                    
                    result = {
                        "operation": "groupby",
                        "group_column": group_col,
                        "aggregation": agg_part,
                        "groups_count": len(grouped),
                        "results": grouped.head(10).to_dict('records')
                    }
                else:
                    return "Group by query should be in format: 'operation by column'"
            
            else:
                return f"Operation '{operation}' not supported. Use 'filter' or 'groupby'"
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return f"Error processing query: {str(e)}"
    
    @staticmethod
    @tool
    def find_correlations(df_json: str, target_column: str = "", threshold: float = 0.5) -> str:
        """Find correlations between variables in the dataset"""
        try:
            df = pd.read_json(io.StringIO(df_json))
            
            # Get only numeric columns
            numeric_df = df.select_dtypes(include=[np.number])
            
            if len(numeric_df.columns) < 2:
                return "Error: Need at least 2 numeric columns to calculate correlations"
            
            # Calculate correlation matrix
            corr_matrix = numeric_df.corr()
            
            results = {
                "correlation_matrix": corr_matrix.to_dict(),
                "strong_correlations": [],
                "summary": {}
            }
            
            # Find strong correlations
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    corr_value = corr_matrix.iloc[i, j]
                    
                    if abs(corr_value) >= threshold:
                        results["strong_correlations"].append({
                            "variable1": col1,
                            "variable2": col2,
                            "correlation": float(corr_value),
                            "strength": "Strong" if abs(corr_value) >= 0.7 else "Moderate"
                        })
            
            # If target column specified, show correlations with that column
            if target_column and target_column in corr_matrix.columns:
                target_correlations = corr_matrix[target_column].drop(target_column).abs().sort_values(ascending=False)
                results["target_correlations"] = {
                    "target": target_column,
                    "correlations": {col: float(val) for col, val in target_correlations.head(10).items()}
                }
            
            results["summary"] = {
                "total_numeric_variables": len(numeric_df.columns),
                "strong_correlations_found": len(results["strong_correlations"]),
                "threshold_used": threshold
            }
            
            return json.dumps(results, indent=2)
            
        except Exception as e:
            return f"Error finding correlations: {str(e)}"

class DataAnalysisAgent:
    """Main agent class that orchestrates the data analysis workflow using Claude"""
    
    def __init__(self, anthropic_api_key: str, model: str = "claude-3-haiku-20240307"):
        self.llm = ChatAnthropic(
            model=model,
            temperature=0.1,
            api_key=anthropic_api_key,
            max_tokens=4000
        )
        
        # Initialize tools
        self.tools = [
            DataAnalysisTools.analyze_dataframe_structure,
            DataAnalysisTools.perform_statistical_analysis,
            DataAnalysisTools.create_visualization,
            DataAnalysisTools.filter_and_query_data,
            DataAnalysisTools.find_correlations
        ]
        
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # System prompt for Claude
        self.system_prompt = """You are Claude, an expert data analyst AI assistant with access to powerful data analysis tools.

Your capabilities include:
1. **Structure Analysis**: Analyze DataFrame structure, data types, missing values, and basic statistics
2. **Statistical Analysis**: Perform descriptive and advanced statistical analysis on numeric and categorical data
3. **Visualizations**: Create various types of plots (histogram, scatter, bar, line, box, violin, correlation heatmaps)
4. **Data Filtering**: Filter data based on conditions and perform group-by operations
5. **Correlation Analysis**: Find relationships between variables and identify strong correlations

**Guidelines for analysis:**
- Always start by understanding the data structure if it's the first interaction
- Provide clear, actionable insights and interpretations
- Suggest appropriate visualizations based on data types and user questions
- When creating visualizations, choose the most appropriate chart type for the data
- Explain statistical findings in plain language
- Point out interesting patterns, outliers, or anomalies
- Always consider the business/practical context of the analysis

**When users ask questions:**
- Break down complex requests into logical steps
- Use multiple tools when necessary to provide comprehensive analysis
- Provide both statistical results and visual insights when appropriate
- Offer suggestions for further analysis or follow-up questions

Be thorough but concise, and always explain your reasoning behind tool choices and interpretations."""

    def analyze_user_request(self, state: DataAnalysisState) -> Dict[str, Any]:
        """Process user request and determine appropriate analysis approach"""
        
        last_message = state.messages[-1] if state.messages else None
        
        if not last_message:
            return {
                "messages": [AIMessage(content="Hello! I'm Claude, your data analysis assistant. Upload a dataset and ask me any questions about it. I can help you explore the structure, create visualizations, perform statistical analysis, and find insights in your data.")]
            }
        
        # Create conversation context
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Add conversation history (keep last 6 messages for context)
        recent_messages = state.messages[-6:] if len(state.messages) > 6 else state.messages
        for msg in recent_messages:
            if isinstance(msg, HumanMessage):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages.append({"role": "assistant", "content": msg.content})
        
        # Get response from Claude
        response = self.llm_with_tools.invoke(messages)
        
        return {"messages": [response]}
    
    def execute_tools(self, state: DataAnalysisState) -> Dict[str, Any]:
        """Execute any tools called by the agent"""
        tool_node = ToolNode(self.tools)
        return tool_node.invoke(state)
    
    def should_continue(self, state: DataAnalysisState) -> str:
        """Determine whether to continue with tool execution or end"""
        last_message = state.messages[-1]
        
        # Check if the last message has tool calls
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        
        return "end"
    
    def create_workflow(self) -> StateGraph:
        """Create the complete data analysis workflow"""
        
        # Define the state schema for LangGraph
        class GraphState(dict):
            messages: Annotated[Sequence[AnyMessage], add_messages]
            dataframe_json: str = ""
        
        # Create workflow
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("agent", self.analyze_user_request)
        workflow.add_node("tools", self.execute_tools)
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            self.should_continue,
            {
                "tools": "tools",
                "end": END
            }
        )
        
        # Add edge from tools back to agent for potential follow-up
        workflow.add_edge("tools", "agent")
        
        # Compile with memory for conversation persistence
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

# Utility functions for the Streamlit app
def get_suggested_analyses(df: pd.DataFrame) -> List[Dict[str, str]]:
    """Generate suggested analysis prompts based on the dataset characteristics"""
    suggestions = []
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Basic structure analysis
    suggestions.append({
        "title": "📋 Dataset Overview",
        "prompt": "Analyze the structure and provide a comprehensive overview of this dataset"
    })
    
    # Statistical analysis
    if numeric_cols:
        suggestions.append({
            "title": "📊 Statistical Summary", 
            "prompt": f"Perform detailed statistical analysis on the numeric columns: {', '.join(numeric_cols[:3])}"
        })
    
    # Visualization suggestions
    if len(numeric_cols) >= 2:
        suggestions.append({
            "title": "🔗 Correlation Analysis",
            "prompt": "Find and visualize correlations between numeric variables"
        })
        
        suggestions.append({
            "title": "📈 Scatter Plot Analysis",
            "prompt": f"Create scatter plots to explore relationships between {numeric_cols[0]} and {numeric_cols[1]}"
        })
    
    if categorical_cols:
        suggestions.append({
            "title": "📊 Category Distribution",
            "prompt": f"Analyze the distribution of {categorical_cols[0]} and create appropriate visualizations"
        })
    
    # Data quality
    suggestions.append({
        "title": "🔍 Data Quality Check",
        "prompt": "Check for missing values, duplicates, and potential data quality issues"
    })
    
    # Advanced insights
    suggestions.append({
        "title": "💡 Key Insights",
        "prompt": "What are the most interesting patterns, trends, or insights in this dataset?"
    })
    
    return suggestions

def format_analysis_results(results: Dict[str, Any]) -> str:
    """Format analysis results for better display in Streamlit"""
    if isinstance(results, str):
        try:
            results = json.loads(results)
        except:
            return results
    
    formatted = ""
    
    if "structure" in results:
        structure = results["structure"]
        formatted += f"**Dataset Structure:**\n"
        formatted += f"- Shape: {structure['shape']['rows']} rows × {structure['shape']['columns']} columns\n"
        formatted += f"- Memory Usage: {structure['memory_usage']}\n"
        formatted += f"- Duplicate Rows: {structure['duplicate_rows']}\n\n"
    
    if "column_types" in results:
        col_types = results["column_types"]
        formatted += f"**Column Types:**\n"
        if col_types["numeric"]:
            formatted += f"- Numeric: {', '.join(col_types['numeric'])}\n"
        if col_types["categorical"]:
            formatted += f"- Categorical: {', '.join(col_types['categorical'])}\n"
        if col_types["datetime"]:
            formatted += f"- DateTime: {', '.join(col_types['datetime'])}\n"
        formatted += "\n"
    
    return formatted if formatted else str(results)