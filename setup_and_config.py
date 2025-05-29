"""
Setup and Configuration for Agentic Data Analysis
This file contains setup utilities and configuration management
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

class Config:
    """Configuration management for the application"""
    
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.model_name = os.getenv("MODEL_NAME", "gpt-4")
        self.temperature = float(os.getenv("TEMPERATURE", "0.1"))
        self.max_tokens = int(os.getenv("MAX_TOKENS", "4000"))
        self.debug_mode = os.getenv("DEBUG_MODE", "False").lower() == "true"
        
    def validate(self):
        """Validate configuration settings"""
        issues = []
        
        if not self.openai_api_key:
            issues.append("OPENAI_API_KEY is not set")
        
        if self.temperature < 0 or self.temperature > 1:
            issues.append("TEMPERATURE must be between 0 and 1")
            
        if self.max_tokens < 100:
            issues.append("MAX_TOKENS should be at least 100")
            
        return issues

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        "streamlit",
        "pandas", 
        "numpy",
        "plotly",
        "langchain",
        "langchain-openai",
        "langchain-core", 
        "langgraph",
        "openai",
        "python-dotenv"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies():
    """Install missing dependencies"""
    missing = check_dependencies()
    
    if missing:
        print(f"Installing missing packages: {', '.join(missing)}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
            print("✅ All dependencies installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing dependencies: {e}")
            return False
    else:
        print("✅ All dependencies are already installed!")
        return True

def create_env_file():
    """Create a sample .env file"""
    env_content = """# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Model Settings
MODEL_NAME=gpt-4
TEMPERATURE=0.1
MAX_TOKENS=4000

# Application Settings
DEBUG_MODE=False

# Streamlit Configuration (optional)
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
"""
    
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ Created .env file. Please update it with your API keys.")
    else:
        print("ℹ️ .env file already exists.")

def setup_project_structure():
    """Create the recommended project structure"""
    directories = [
        "data",
        "logs", 
        "exports",
        "temp"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        
        # Create .gitkeep files to ensure directories are tracked by git
        gitkeep_path = Path(directory) / ".gitkeep"
        if not gitkeep_path.exists():
            gitkeep_path.touch()
    
    print("✅ Project structure created!")

def create_gitignore():
    """Create a comprehensive .gitignore file"""
    gitignore_content = """# Environment variables
.env
.env.local
.env.production

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Logs
logs/
*.log

# Data files (add specific patterns as needed)
data/*.csv
data/*.xlsx
data/*.json
temp/

# Streamlit
.streamlit/

# OS
.DS_Store
Thumbs.db

# Application specific
exports/
*.pkl
*.joblib
"""
    
    if not os.path.exists('.gitignore'):
        with open('.gitignore', 'w') as f:
            f.write(gitignore_content)
        print("✅ Created .gitignore file.")
    else:
        print("ℹ️ .gitignore file already exists.")

def create_readme():
    """Create a comprehensive README.md file"""
    readme_content = """# 🤖 Agentic Data Analysis

An intelligent data analysis assistant built with LangGraph, OpenAI, and Streamlit. Upload your datasets and interact with them using natural language to get insights, visualizations, and statistical analysis.

## ✨ Features

- **🧠 AI-Powered Analysis**: Uses advanced LLM agents to understand and analyze your data
- **📊 Smart Visualizations**: Automatically creates appropriate charts and graphs
- **📈 Statistical Insights**: Performs comprehensive statistical analysis
- **🔍 Data Exploration**: Interactive data filtering and querying
- **💬 Natural Language Interface**: Ask questions about your data in plain English
- **🎯 Correlation Analysis**: Finds relationships between variables
- **📋 Automated Reports**: Generates comprehensive analysis reports

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd AgenticDataAnalysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment:
```bash
python setup_and_config.py
```

4. Update the `.env` file with your OpenAI API key:
```bash
OPENAI_API_KEY=your_actual_api_key_here
```

5. Run the application:
```bash
streamlit run streamlit_app_simplified.py --server.maxUploadSize 2000
```

## 📖 Usage

1. **Upload Data**: Use the sidebar to upload a CSV file
2. **Configure API**: Enter your OpenAI API key in the sidebar
3. **Start Analyzing**: Use the chat interface to ask questions about your data
4. **Explore**: Try the suggested analysis prompts or ask custom questions

### Example Questions

- "What's the structure of this dataset?"
- "Show me statistical summaries of all numeric columns"
- "Create a correlation heatmap"
- "Find outliers in the data"
- "What are the most interesting insights?"
- "Create a scatter plot of X vs Y"
- "Filter data where column > value"

## 🏗️ Architecture

The application uses a modular architecture with:

- **Agent Nodes** (`agent_nodes.py`): Core AI agent logic and tools
- **Streamlit App** (`streamlit_app_simplified.py`): User interface
- **Configuration** (`setup_and_config.py`): Setup and configuration utilities

### Key Components

1. **DataAnalysisAgent**: Main orchestrator for analysis workflows
2. **DataAnalysisTools**: Collection of specialized analysis tools
3. **LangGraph Workflow**: Stateful conversation management
4. **Streamlit Interface**: Interactive web application

## 🛠️ Tools Available

- **Structure Analysis**: DataFrame inspection and profiling
- **Statistical Analysis**: Descriptive and advanced statistics
- **Visualization**: Multiple chart types (histogram, scatter, bar, line, box, violin, correlation)
- **Data Filtering**: Conditional filtering and group-by operations
- **Correlation Analysis**: Find relationships between variables

## 📁 Project Structure

```
AgenticDataAnalysis/
├── agent_nodes.py              # Core agent logic and tools
├── streamlit_app_simplified.py # Main Streamlit application
├── setup_and_config.py        # Setup and configuration utilities  
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (create this)
├── README.md                   # This file
├── data/                       # Data files directory
├── logs/                       # Application logs
├── exports/                    # Exported analysis results
└── temp/                       # Temporary files
```

## 🔧 Configuration

Key settings in `.env`:

- `OPENAI_API_KEY`: Your OpenAI API key
- `MODEL_NAME`: AI model to use (gpt-4, gpt-3.5-turbo)
- `TEMPERATURE`: Response creativity (0.0-1.0)
- `MAX_TOKENS`: Maximum response length
- `DEBUG_MODE`: Enable debug logging

## 🚨 Troubleshooting

### Common Issues

1. **API Key Error**: Ensure your OpenAI API key is valid and has sufficient credits
2. **Memory Issues**: For large datasets, consider sampling or using more powerful hardware
3. **Timeout Errors**: Complex analyses may take time; be patient or break into smaller questions

### Getting Help

- Check the logs in the `logs/` directory
- Enable `DEBUG_MODE=True` in your `.env` file for detailed logging
- Ensure all dependencies are installed correctly

## 📊 Sample Datasets

For testing, you can download sample datasets from:
- [Kaggle Fraud Detection Dataset](https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets/data)
- Any CSV file with mixed data types works well

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [LangGraph](https://python.langchain.com/docs/langgraph) for agent orchestration
- Powered by [OpenAI](https://openai.com/) for language understanding
- UI created with [Streamlit](https://streamlit.io/)
- Visualizations using [Plotly](https://plotly.com/python/)

## 📈 Roadmap

- [ ] Support for more file formats (Excel, JSON, Parquet)
- [ ] Advanced ML model integration
- [ ] Report export functionality  
- [ ] Multi-dataset comparison
- [ ] Real-time data streaming support
- [ ] Custom visualization templates

---

**Happy Analyzing! 🎉**
"""
    
    if not os.path.exists('README.md'):
        with open('README.md', 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print("✅ Created README.md file.")
    else:
        print("ℹ️ README.md file already exists.")

def setup_streamlit_config():
    """Create Streamlit configuration"""
    streamlit_dir = Path(".streamlit")
    streamlit_dir.mkdir(exist_ok=True)
    
    config_content = """[general]
dataFrameSerialization = "legacy"

[server]
maxUploadSize = 2000
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
"""
    
    config_path = streamlit_dir / "config.toml"
    if not config_path.exists():
        with open(config_path, 'w') as f:
            f.write(config_content)
        print("✅ Created Streamlit configuration.")
    else:
        print("ℹ️ Streamlit configuration already exists.")

def run_setup():
    """Run the complete setup process"""
    print("🚀 Setting up Agentic Data Analysis...")
    print("=" * 50)
    
    # Check and install dependencies
    print("\n1. Checking dependencies...")
    if not install_dependencies():
        print("❌ Setup failed during dependency installation.")
        return False
    
    # Create project structure
    print("\n2. Creating project structure...")
    setup_project_structure()
    
    # Create configuration files
    print("\n3. Creating configuration files...")
    create_env_file()
    create_gitignore()
    create_readme()
    setup_streamlit_config()
    
    # Validate configuration
    print("\n4. Validating configuration...")
    config = Config()
    issues = config.validate()
    
    if issues:
        print("⚠️ Configuration issues found:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nPlease update your .env file and run the setup again.")
    else:
        print("✅ Configuration looks good!")
    
    print("\n" + "=" * 50)
    print("🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Update your .env file with the correct API keys")
    print("2. Run: streamlit run streamlit_app_simplified.py --server.maxUploadSize 2000")
    print("3. Upload a CSV file and start analyzing!")
    
    return True

def check_system_requirements():
    """Check system requirements and compatibility"""
    print("🔍 Checking system requirements...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    else:
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check available memory (rough estimate)
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb < 4:
            print(f"⚠️ Low memory detected ({memory_gb:.1f}GB). Consider using smaller datasets.")
        else:
            print(f"✅ Available memory: {memory_gb:.1f}GB")
    except ImportError:
        print("ℹ️ Could not check memory (psutil not available)")
    
    return True

if __name__ == "__main__":
    print("🤖 Agentic Data Analysis Setup")
    print("===============================")
    
    if check_system_requirements():
        run_setup()
    else:
        print("❌ System requirements not met. Please upgrade your Python version.")
        sys.exit(1)