# 🤖 Agentic Data Analysis

An intelligent multi-modal AI system for comprehensive data analysis and document processing using LangGraph, Claude AI, and Streamlit.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![Claude](https://img.shields.io/badge/claude-AI-purple.svg)
![LangGraph](https://img.shields.io/badge/langgraph-enabled-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

## ✨ Features

### 🧠 Multi-Agent Architecture
- **Data Analysis Agent** - Specialized in statistical analysis and data profiling
- **Visualization Agent** - Creates intelligent charts and graphs
- **Insight Agent** - Generates business insights and recommendations
- **Report Agent** - Compiles comprehensive analysis reports
- **Supervisor Agent** - Coordinates the entire workflow

### 📊 Data Analysis Capabilities
- **Automatic Data Profiling** - Structure analysis, data types, missing values
- **Statistical Analysis** - Descriptive statistics, correlations, distributions
- **Smart Visualizations** - Auto-generated charts based on data characteristics
- **Outlier Detection** - Identifies anomalies and unusual patterns
- **Pattern Recognition** - Discovers hidden relationships in data

### 📄 Document Processing (RAG)
- **PDF Text Extraction** - Extract content from any PDF document
- **Vector Embeddings** - Semantic search through documents
- **Question Answering** - Chat with your documents using natural language
- **Source References** - Exact citations for every answer
- **Multi-Document Support** - Analyze multiple files simultaneously

### 🔗 Multi-Modal Analysis
- **Cross-Reference Analysis** - Find connections between documents and data
- **Integrated Insights** - Combined analysis of PDFs and CSV data
- **Comprehensive Reporting** - Unified analysis across all data sources

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Anthropic API key (for Claude AI)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/sandeepgoriparthi/AgenticDataAnalysis.git
cd AgenticDataAnalysis
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**
```bash
cp .env.example .env
# Edit .env file with your API keys
```

4. **Run the application:**
```bash
streamlit run streamlit_app_simplified.py --server.maxUploadSize 2000
```

## 🎯 Available Applications

| Application | Description | Use Case |
|-------------|-------------|----------|
| `streamlit_app_simplified.py` | Main multi-agent application | General data analysis with Claude |
| `working_data_analyst.py` | Automatic data analysis | Quick data profiling and insights |
| `advanced_agentic_system.py` | Multi-agent workflow | Complex analysis with agent coordination |
| `rag_system.py` | Document Q&A system | PDF analysis and question answering |
| `pdf_analysis_agent.py` | PDF + CSV analysis | Multi-modal document and data analysis |

## 📊 Usage Examples

### 1. Data Analysis
Upload a CSV file and watch the AI automatically:
- Analyze data structure and quality
- Generate statistical summaries
- Create intelligent visualizations
- Provide actionable insights

### 2. Document Q&A (RAG)
Upload PDF documents and ask questions like:
- "What are the main findings?"
- "Who are the key stakeholders?"
- "What recommendations are made?"

### 3. Multi-Modal Analysis
Upload both CSV and PDF files to find connections between document content and data elements.

## 🔧 Configuration

### Environment Variables (.env)
```bash
# Anthropic/Claude Configuration
ANTHROPIC_API_KEY=sk-ant-your-api-key-here

# Model Settings
MODEL_NAME=claude-3-haiku-20240307
TEMPERATURE=0.1
MAX_TOKENS=4000
```

## 📁 Project Structure

```
AgenticDataAnalysis/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── .env.example                        # Environment variables template
├── .gitignore                          # Git ignore rules
├── LICENSE                             # MIT License
├── 
├── 📊 Core Applications
├── streamlit_app_simplified.py         # Main Streamlit app (Claude-based)
├── working_data_analyst.py             # Working data analysis demo
├── advanced_agentic_system.py          # Multi-agent system
├── pdf_analysis_agent.py               # PDF processing system
├── rag_system.py                       # True RAG implementation
├── 
├── 🧠 Agent Components  
├── agent_nodes.py                      # Core agent logic and tools
├── 
├── 🔧 Utilities
├── setup_and_config.py                 # Setup and configuration
├── usage_examples.py                   # Demo data and examples
├── test_claude.py                      # API key tester
├── 
├── 📁 Data & Logs
├── data/                               # Data files directory
├── logs/                               # Application logs
├── exports/                            # Exported results
└── temp/                               # Temporary files
```

## 🛠️ Technical Architecture

### Multi-Agent System
Built using **LangGraph** with specialized agents:
- **Supervisor Agent** - Orchestrates workflow
- **Analysis Agent** - Performs data analysis
- **Visualization Agent** - Creates charts
- **Insight Agent** - Generates findings
- **Report Agent** - Compiles results

### RAG Implementation
- **Vector Embeddings** using HuggingFace transformers
- **FAISS Vector Database** for semantic search
- **Document Chunking** with LangChain splitters
- **Context Retrieval** for accurate Q&A

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[LangChain](https://langchain.com/)** - For the agent framework
- **[Anthropic](https://anthropic.com/)** - For Claude AI
- **[Streamlit](https://streamlit.io/)** - For the web interface
- **[Plotly](https://plotly.com/)** - For visualizations

## 🚨 Troubleshooting

### Common Issues

1. **API Key Error**: Ensure your Anthropic API key is valid and has credits
2. **Memory Issues**: For large datasets, consider using smaller samples
3. **Upload Limits**: Streamlit's default upload limit is 200MB

### Getting Help

- Check the [Issues](https://github.com/sandeepgoriparthi/AgenticDataAnalysis/issues) page
- Enable `DEBUG_MODE=True` in your `.env` file for detailed logging

## 📈 Roadmap

- [ ] Support for more file formats (Excel, JSON, Parquet)
- [ ] Advanced ML model integration
- [ ] Real-time data streaming
- [ ] Custom visualization templates
- [ ] Multi-language support
- [ ] API endpoint for programmatic access

---

**Built with ❤️ using LangGraph, Claude AI, and Streamlit**

⭐ **If you find this project helpful, please consider giving it a star!**

## 🎬 Demo

Try the live application:
1. Clone this repository
2. Install dependencies
3. Add your Anthropic API key
4. Run `streamlit run streamlit_app_simplified.py`
5. Upload CSV or PDF files and explore!
