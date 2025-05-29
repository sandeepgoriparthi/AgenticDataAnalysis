"""
PDF Document Analysis Agent
Multi-modal AI system that can analyze both PDFs and CSV data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Annotated, Sequence, TypedDict, Union
import json
import io
from datetime import datetime
import PyPDF2
import pdfplumber
from PIL import Image
import base64

# LangGraph and LangChain imports
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Page configuration
st.set_page_config(
    page_title="📄 PDF & Data Analysis Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced State for PDF and Data Analysis
class MultiModalState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    document_content: str
    document_type: str  # "pdf" or "csv" or "both"
    dataframe_json: str
    analysis_results: Dict[str, Any]
    document_summary: str
    key_insights: List[str]
    report: str
    next_agent: str

# PDF Processing Tools
@tool
def extract_pdf_content(pdf_base64: str) -> str:
    """Extract text content from PDF file"""
    try:
        # Decode base64 PDF
        pdf_bytes = base64.b64decode(pdf_base64)
        
        # Try pdfplumber first (better for structured text)
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                full_text = ""
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        full_text += f"\n--- Page {page_num + 1} ---\n{text}\n"
                
                if full_text.strip():
                    return full_text
        except:
            pass
        
        # Fallback to PyPDF2
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            full_text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text:
                    full_text += f"\n--- Page {page_num + 1} ---\n{text}\n"
            
            return full_text if full_text.strip() else "Could not extract readable text from PDF"
        except Exception as e:
            return f"Error extracting PDF content: {str(e)}"
            
    except Exception as e:
        return f"Error processing PDF: {str(e)}"

@tool
def analyze_document_structure(content: str) -> str:
    """Analyze the structure and content of a document"""
    try:
        if not content or len(content.strip()) < 10:
            return "Document appears to be empty or very short"
        
        lines = content.split('\n')
        words = content.split()
        
        # Basic statistics
        analysis = {
            "document_stats": {
                "total_characters": len(content),
                "total_words": len(words),
                "total_lines": len(lines),
                "average_words_per_line": round(len(words) / max(len(lines), 1), 2),
                "estimated_reading_time": round(len(words) / 200, 1)  # 200 words per minute
            },
            "content_analysis": {
                "has_headers": any("---" in line for line in lines[:20]),
                "has_numbers": any(char.isdigit() for char in content),
                "has_dates": any(word for word in words if any(date_term in word.lower() for date_term in ['2020', '2021', '2022', '2023', '2024', '2025'])),
                "language_indicators": {
                    "english_common_words": sum(1 for word in words[:100] if word.lower() in ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']),
                }
            }
        }
        
        # Find potential sections/headers
        potential_headers = []
        for line in lines[:50]:  # Check first 50 lines
            line = line.strip()
            if line and (line.isupper() or line.endswith(':') or len(line.split()) <= 6):
                if not line.isdigit() and len(line) > 2:
                    potential_headers.append(line)
        
        analysis["potential_sections"] = potential_headers[:10]
        
        # Extract key terms (simple approach)
        word_freq = {}
        for word in words:
            word = word.lower().strip('.,!?;:"()[]{}')
            if len(word) > 3 and word.isalpha():
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top 20 most frequent meaningful words
        common_words = {'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 'were', 'said', 'each', 'which', 'their', 'time', 'would', 'there', 'could', 'other'}
        filtered_words = {k: v for k, v in word_freq.items() if k not in common_words and v > 1}
        top_terms = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)[:20]
        
        analysis["key_terms"] = [{"term": term, "frequency": freq} for term, freq in top_terms]
        
        return json.dumps(analysis, indent=2)
        
    except Exception as e:
        return f"Error analyzing document structure: {str(e)}"

@tool
def extract_key_information(content: str, focus_area: str = "general") -> str:
    """Extract key information from document content based on focus area"""
    try:
        lines = content.split('\n')
        key_info = {
            "focus_area": focus_area,
            "extracted_information": []
        }
        
        if focus_area == "financial":
            # Look for financial information
            financial_keywords = ['revenue', 'profit', 'cost', 'budget', 'sales', 'income', 'expense', 'roi', 'margin']
            for line in lines:
                if any(keyword in line.lower() for keyword in financial_keywords):
                    if any(char.isdigit() for char in line):
                        key_info["extracted_information"].append(line.strip())
        
        elif focus_area == "dates":
            # Extract date-related information
            date_keywords = ['date', 'year', 'month', 'quarter', 'deadline', 'schedule']
            for line in lines:
                if any(keyword in line.lower() for keyword in date_keywords):
                    key_info["extracted_information"].append(line.strip())
        
        elif focus_area == "people":
            # Extract names and people-related info
            people_keywords = ['name', 'author', 'director', 'manager', 'ceo', 'president', 'contact']
            for line in lines:
                if any(keyword in line.lower() for keyword in people_keywords):
                    key_info["extracted_information"].append(line.strip())
        
        else:  # general
            # Extract important-looking lines
            important_indicators = ['important', 'key', 'summary', 'conclusion', 'result', 'finding']
            for line in lines:
                line_clean = line.strip()
                if line_clean and (
                    any(indicator in line_clean.lower() for indicator in important_indicators) or
                    line_clean.endswith(':') or
                    line_clean.isupper() and len(line_clean) < 100
                ):
                    key_info["extracted_information"].append(line_clean)
        
        # Limit to top 15 most relevant items
        key_info["extracted_information"] = key_info["extracted_information"][:15]
        
        return json.dumps(key_info, indent=2)
        
    except Exception as e:
        return f"Error extracting key information: {str(e)}"

@tool
def compare_document_with_data(doc_content: str, df_json: str) -> str:
    """Compare document content with CSV data to find relationships"""
    try:
        df = pd.read_json(io.StringIO(df_json))
        
        # Get column names and sample values from CSV
        csv_elements = {
            "columns": list(df.columns),
            "sample_values": {}
        }
        
        for col in df.columns:
            if df[col].dtype == 'object':
                csv_elements["sample_values"][col] = df[col].dropna().unique()[:10].tolist()
            else:
                csv_elements["sample_values"][col] = [df[col].min(), df[col].max(), df[col].mean()]
        
        # Look for mentions of CSV elements in document
        doc_lower = doc_content.lower()
        relationships = {
            "column_mentions": [],
            "value_matches": [],
            "potential_connections": []
        }
        
        # Check for column name mentions
        for col in csv_elements["columns"]:
            if col.lower() in doc_lower:
                relationships["column_mentions"].append(col)
        
        # Check for value mentions (for categorical data)
        for col, values in csv_elements["sample_values"].items():
            if isinstance(values, list) and all(isinstance(v, str) for v in values):
                for value in values:
                    if isinstance(value, str) and len(value) > 2 and value.lower() in doc_lower:
                        relationships["value_matches"].append({"column": col, "value": value})
        
        # Generate insights about relationships
        if relationships["column_mentions"] or relationships["value_matches"]:
            relationships["potential_connections"].append("Document contains references to data elements")
        
        if len(relationships["column_mentions"]) > len(csv_elements["columns"]) * 0.3:
            relationships["potential_connections"].append("Strong correlation between document and dataset")
        
        return json.dumps(relationships, indent=2)
        
    except Exception as e:
        return f"Error comparing document with data: {str(e)}"

# Specialized Agents for Multi-Modal Analysis
class DocumentAnalysisAgent:
    def __init__(self, llm):
        self.llm = llm
        self.tools = [extract_pdf_content, analyze_document_structure, extract_key_information]
        self.llm_with_tools = llm.bind_tools(self.tools)
    
    def analyze(self, state: MultiModalState) -> Dict[str, Any]:
        system_prompt = """You are a Document Analysis Agent specialized in analyzing PDF documents and extracting key information.

        Your tasks:
        1. Extract and analyze document structure
        2. Identify key information and insights
        3. Summarize main points
        4. Extract specific data based on document type

        Available tools:
        - extract_pdf_content: Extract text from PDF files
        - analyze_document_structure: Analyze document organization and stats
        - extract_key_information: Extract specific types of information

        Always provide comprehensive analysis with clear insights.
        """
        
        task_description = f"Analyze the uploaded document. Type: {state.get('document_type', 'unknown')}"
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=task_description)
        ]
        
        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response], "next_agent": "insight"}

class MultiModalInsightAgent:
    def __init__(self, llm):
        self.llm = llm
        self.tools = [compare_document_with_data] if hasattr(self, 'has_both_data') else []
        self.llm_with_tools = llm.bind_tools(self.tools) if self.tools else llm
    
    def generate_insights(self, state: MultiModalState) -> Dict[str, Any]:
        system_prompt = """You are a Multi-Modal Insight Agent. Generate comprehensive insights from documents and/or data.

        Based on the analysis results, provide:
        1. Key findings from the document(s)
        2. Important insights and patterns
        3. Summary of main points
        4. Actionable recommendations
        5. If both PDF and CSV data are available, find connections between them

        Make your insights clear, structured, and valuable.
        """
        
        doc_type = state.get('document_type', 'unknown')
        analysis_context = f"Document type: {doc_type}. Generate comprehensive insights."
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=analysis_context)
        ]
        
        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response], "next_agent": "report"}

class MultiModalReportAgent:
    def __init__(self, llm):
        self.llm = llm
    
    def generate_report(self, state: MultiModalState) -> Dict[str, Any]:
        system_prompt = """You are a Report Generation Agent for multi-modal analysis.

        Create a comprehensive report with:
        1. Executive Summary
        2. Document Analysis (if PDF uploaded)
        3. Data Analysis (if CSV uploaded) 
        4. Key Findings and Insights
        5. Cross-Modal Connections (if both types available)
        6. Recommendations and Next Steps
        7. Conclusion

        Make it professional, well-structured, and actionable.
        """
        
        doc_type = state.get('document_type', 'unknown')
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Generate comprehensive report for {doc_type} analysis.")
        ]
        
        response = self.llm.invoke(messages)
        return {"messages": [response], "report": response.content, "next_agent": "end"}

# Create Multi-Modal Workflow
def create_multimodal_workflow(api_key: str):
    """Create workflow that handles both PDFs and CSV data"""
    
    llm = ChatAnthropic(
        model="claude-3-haiku-20240307",
        temperature=0.1,
        api_key=api_key,
        max_tokens=3000
    )
    
    # Initialize agents
    doc_agent = DocumentAnalysisAgent(llm)
    insight_agent = MultiModalInsightAgent(llm)
    report_agent = MultiModalReportAgent(llm)
    
    # Import data analysis tools from previous version
    from working_data_analyst import analyze_data_structure, perform_statistical_analysis
    
    # Create workflow
    workflow = StateGraph(MultiModalState)
    
    # Add nodes
    workflow.add_node("document_agent", doc_agent.analyze)
    workflow.add_node("insight_agent", insight_agent.generate_insights)
    workflow.add_node("report_agent", report_agent.generate_report)
    workflow.add_node("tools", ToolNode([
        extract_pdf_content,
        analyze_document_structure,
        extract_key_information,
        compare_document_with_data
    ]))
    
    # Routing logic
    def route_next(state: MultiModalState) -> str:
        last_message = state["messages"][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        
        next_agent = state.get("next_agent", "insight")
        if next_agent == "end":
            return "end"
        elif next_agent == "insight":
            return "insight_agent"
        elif next_agent == "report":
            return "report_agent"
        else:
            return "end"
    
    # Set entry point
    workflow.set_entry_point("document_agent")
    
    # Add edges
    workflow.add_conditional_edges(
        "document_agent",
        route_next,
        {
            "tools": "tools",
            "insight_agent": "insight_agent",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "insight_agent",
        route_next,
        {
            "tools": "tools",
            "report_agent": "report_agent",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "report_agent",
        route_next,
        {"end": END}
    )
    
    workflow.add_conditional_edges(
        "tools",
        route_next,
        {
            "document_agent": "document_agent",
            "insight_agent": "insight_agent",
            "report_agent": "report_agent",
            "end": END
        }
    )
    
    return workflow.compile()

# Streamlit App
def main():
    st.title("📄 Multi-Modal AI Analysis Agent")
    st.markdown("**Analyze PDFs, CSV Data, or Both Together!**")
    
    # Custom CSS
    st.markdown("""
    <style>
        .upload-section {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            margin: 1rem 0;
        }
        .analysis-type {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            margin: 0.5rem 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'workflow' not in st.session_state:
        st.session_state.workflow = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = {"pdf": None, "csv": None}
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            help="Required for Claude AI analysis"
        )
        
        if api_key:
            if st.session_state.workflow is None:
                with st.spinner("🤖 Initializing Multi-Modal AI System..."):
                    st.session_state.workflow = create_multimodal_workflow(api_key)
                st.success("✅ AI Agents Ready!")
        
        st.header("📁 File Upload")
        
        # PDF Upload
        pdf_file = st.file_uploader(
            "📄 Upload PDF Document",
            type=['pdf'],
            help="Upload PDF for document analysis"
        )
        
        if pdf_file:
            st.session_state.uploaded_files["pdf"] = pdf_file
            st.success(f"📄 PDF uploaded: {pdf_file.name}")
        
        # CSV Upload
        csv_file = st.file_uploader(
            "📊 Upload CSV Data",
            type=['csv'],
            help="Upload CSV for data analysis"
        )
        
        if csv_file:
            st.session_state.uploaded_files["csv"] = csv_file
            df = pd.read_csv(csv_file)
            st.success(f"📊 CSV uploaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Main Content
    if st.session_state.workflow and (st.session_state.uploaded_files["pdf"] or st.session_state.uploaded_files["csv"]):
        
        # Analysis Type Selection
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("### 🎯 Select Analysis Type")
        
        analysis_types = []
        if st.session_state.uploaded_files["pdf"]:
            analysis_types.append("📄 PDF Document Analysis")
        if st.session_state.uploaded_files["csv"]:
            analysis_types.append("📊 CSV Data Analysis")
        if st.session_state.uploaded_files["pdf"] and st.session_state.uploaded_files["csv"]:
            analysis_types.append("🔗 Combined PDF + CSV Analysis")
        
        selected_analysis = st.selectbox("Choose analysis type:", analysis_types)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Start Analysis Button
        if st.button("🚀 Start AI Analysis", type="primary", use_container_width=True):
            with st.spinner("🤖 AI Agents are analyzing your files..."):
                try:
                    # Prepare data based on selection
                    document_content = ""
                    dataframe_json = ""
                    document_type = ""
                    
                    if "PDF" in selected_analysis and st.session_state.uploaded_files["pdf"]:
                        # Convert PDF to base64
                        pdf_base64 = base64.b64encode(st.session_state.uploaded_files["pdf"].getvalue()).decode()
                        document_type = "pdf"
                        
                    if "CSV" in selected_analysis and st.session_state.uploaded_files["csv"]:
                        df = pd.read_csv(st.session_state.uploaded_files["csv"])
                        dataframe_json = df.to_json()
                        if document_type:
                            document_type = "both"
                        else:
                            document_type = "csv"
                    
                    # Run analysis
                    initial_state = {
                        "messages": [HumanMessage(content=f"Perform {selected_analysis}")],
                        "document_content": pdf_base64 if "PDF" in selected_analysis else "",
                        "document_type": document_type,
                        "dataframe_json": dataframe_json,
                        "analysis_results": {},
                        "document_summary": "",
                        "key_insights": [],
                        "report": "",
                        "next_agent": "document"
                    }
                    
                    result = st.session_state.workflow.invoke(initial_state)
                    st.session_state.analysis_results = result
                    
                except Exception as e:
                    st.error(f"Analysis error: {str(e)}")
    
    # Display Results
    if st.session_state.analysis_results:
        st.success("✅ Analysis Complete!")
        
        # Agent Communications
        st.header("🤖 AI Agent Analysis")
        for i, message in enumerate(st.session_state.analysis_results.get("messages", [])):
            if isinstance(message, AIMessage):
                with st.expander(f"Agent Response {i+1} - {message.content[:100]}..."):
                    st.markdown(message.content)
        
        # Final Report
        if st.session_state.analysis_results.get("report"):
            st.header("📋 Comprehensive Analysis Report")
            st.markdown(st.session_state.analysis_results["report"])
        
        # Download Report
        if st.session_state.analysis_results.get("report"):
            st.download_button(
                label="💾 Download Analysis Report",
                data=st.session_state.analysis_results["report"],
                file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
    
    else:
        # Welcome Screen
        st.markdown("### 🎯 What This System Can Do:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="analysis-type">
                <h4>📄 PDF Analysis</h4>
                <p>• Extract text content<br>
                • Analyze document structure<br>
                • Identify key information<br>
                • Generate summaries<br>
                • Find important sections</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="analysis-type">
                <h4>📊 CSV Analysis</h4>
                <p>• Statistical analysis<br>
                • Data visualization<br>
                • Pattern detection<br>
                • Correlation analysis<br>
                • Outlier identification</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="analysis-type">
                <h4>🔗 Combined Analysis</h4>
                <p>• Cross-reference data<br>
                • Find connections<br>
                • Validate information<br>
                • Comprehensive insights<br>
                • Integrated reporting</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.info("🔧 **Setup Instructions:**\n1. Add your Anthropic API key\n2. Upload PDF and/or CSV files\n3. Select analysis type\n4. Start AI analysis!")

if __name__ == "__main__":
    main()