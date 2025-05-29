"""
Usage Examples and Demo Script for Agentic Data Analysis
This file contains examples of how to use the data analysis agent
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def create_sample_dataset(filename="sample_data.csv", num_rows=1000):
    """Create a sample dataset for testing the agent"""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Generate sample data
    data = {
        'customer_id': range(1, num_rows + 1),
        'age': np.random.normal(35, 12, num_rows).astype(int),
        'income': np.random.lognormal(10, 0.5, num_rows).astype(int),
        'spending_score': np.random.randint(1, 101, num_rows),
        'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books', 'Sports'], num_rows),
        'purchase_amount': np.random.exponential(50, num_rows).round(2),
        'is_premium': np.random.choice([True, False], num_rows, p=[0.3, 0.7]),
        'satisfaction_score': np.random.normal(7.5, 1.5, num_rows).round(1),
        'purchase_date': [
            datetime.now() - timedelta(days=random.randint(1, 365)) 
            for _ in range(num_rows)
        ]
    }
    
    # Add some correlations to make the data more interesting
    # Higher income generally leads to higher spending
    for i in range(num_rows):
        if data['income'][i] > 50000:
            data['spending_score'][i] = min(100, data['spending_score'][i] + random.randint(10, 30))
            data['purchase_amount'][i] *= random.uniform(1.2, 2.0)
        
        # Premium customers have higher satisfaction
        if data['is_premium'][i]:
            data['satisfaction_score'][i] = min(10, data['satisfaction_score'][i] + random.uniform(0.5, 2.0))
        
        # Ensure age is reasonable
        data['age'][i] = max(18, min(80, data['age'][i]))
        
        # Ensure satisfaction is within bounds
        data['satisfaction_score'][i] = max(1, min(10, data['satisfaction_score'][i]))
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some missing values to make it realistic
    missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
    df.loc[missing_indices, 'satisfaction_score'] = np.nan
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"✅ Created sample dataset: {filename}")
    print(f"   - {len(df)} rows, {len(df.columns)} columns")
    print(f"   - Columns: {', '.join(df.columns)}")
    
    return df

class AgentUsageExamples:
    """Collection of example queries and expected behaviors"""
    
    @staticmethod
    def basic_analysis_examples():
        """Basic data analysis examples"""
        return [
            {
                "category": "Data Structure",
                "examples": [
                    "Analyze the structure of this dataset",
                    "What columns do I have and what are their data types?",
                    "How many rows and columns are in my data?",
                    "Are there any missing values?",
                    "Show me a summary of the dataset"
                ]
            },
            {
                "category": "Statistical Analysis", 
                "examples": [
                    "Perform statistical analysis on all numeric columns",
                    "What are the mean, median, and standard deviation of income?",
                    "Show me descriptive statistics for spending_score",
                    "Are there any outliers in the purchase_amount column?",
                    "Calculate quartiles for age distribution"
                ]
            },
            {
                "category": "Data Filtering",
                "examples": [
                    "Filter customers where age > 40",
                    "Show me premium customers only",
                    "Find customers with income > 75000",
                    "Filter data where satisfaction_score >= 8",
                    "Show customers who purchased Electronics"
                ]
            }
        ]
    
    @staticmethod
    def visualization_examples():
        """Visualization examples"""
        return [
            {
                "category": "Distribution Plots",
                "examples": [
                    "Create a histogram of age distribution",
                    "Show the distribution of income",
                    "Plot spending_score distribution",
                    "Create a histogram of purchase_amount",
                    "Show satisfaction_score distribution"
                ]
            },
            {
                "category": "Relationship Plots",
                "examples": [
                    "Create a scatter plot of income vs spending_score",
                    "Plot age vs purchase_amount",
                    "Show the relationship between income and satisfaction_score",
                    "Create a scatter plot colored by category",
                    "Plot spending_score vs satisfaction_score"
                ]
            },
            {
                "category": "Categorical Analysis",
                "examples": [
                    "Create a bar chart of category distribution",
                    "Show premium vs non-premium customer counts",
                    "Plot average income by category",
                    "Create a bar chart of satisfaction_score by category",
                    "Show spending patterns by customer type"
                ]
            },
            {
                "category": "Advanced Visualizations",
                "examples": [
                    "Create a correlation heatmap",
                    "Show box plots of income by category", 
                    "Create violin plots for spending_score",
                    "Plot time series of purchases over time",
                    "Show correlation between all numeric variables"
                ]
            }
        ]
    
    @staticmethod
    def insight_examples():
        """Examples for finding insights"""
        return [
            {
                "category": "Business Insights",
                "examples": [
                    "What are the key insights from this customer data?",
                    "Which customer segments are most valuable?",
                    "How does income relate to spending behavior?",
                    "What factors influence customer satisfaction?",
                    "Are premium customers significantly different?"
                ]
            },
            {
                "category": "Correlation Analysis",
                "examples": [
                    "Find correlations between all variables",
                    "Which variables are most correlated with satisfaction_score?",
                    "Show me strong correlations (>0.5)",
                    "How does age correlate with other variables?",
                    "Find relationships between spending and demographics"
                ]
            },
            {
                "category": "Segmentation Analysis",
                "examples": [
                    "Group customers by age ranges and analyze spending",
                    "Compare premium vs regular customers",
                    "Analyze spending patterns by product category",
                    "Segment customers by income levels",
                    "Find high-value customer characteristics"
                ]
            }
        ]
    
    @staticmethod
    def advanced_examples():
        """Advanced analysis examples"""
        return [
            {
                "category": "Statistical Tests",
                "examples": [
                    "Is there a significant difference in satisfaction between premium and regular customers?",
                    "Test if income distribution is normal",
                    "Compare spending across different categories",
                    "Analyze variance in purchase amounts",
                    "Test correlation significance"
                ]
            },
            {
                "category": "Data Quality",
                "examples": [
                    "Check for data quality issues",
                    "Find and analyze outliers",
                    "Identify duplicate records",
                    "Analyze missing value patterns",
                    "Suggest data cleaning steps"
                ]
            },
            {
                "category": "Predictive Insights",
                "examples": [
                    "What predicts high customer satisfaction?",
                    "Which customers are likely to be high spenders?",
                    "Identify factors that influence purchase amounts",
                    "Find patterns in customer behavior",
                    "Suggest customer retention strategies"
                ]
            }
        ]

def demonstrate_agent_capabilities():
    """Demonstrate the agent's capabilities with example queries"""
    
    print("🤖 Agentic Data Analysis - Capability Demonstration")
    print("=" * 55)
    
    examples = AgentUsageExamples()
    
    # Basic Analysis
    print("\n📊 BASIC DATA ANALYSIS")
    print("-" * 25)
    for category in examples.basic_analysis_examples():
        print(f"\n{category['category']}:")
        for i, example in enumerate(category['examples'][:3], 1):
            print(f"  {i}. {example}")
    
    # Visualizations
    print("\n📈 VISUALIZATION EXAMPLES")
    print("-" * 25)
    for category in examples.visualization_examples():
        print(f"\n{category['category']}:")
        for i, example in enumerate(category['examples'][:3], 1):
            print(f"  {i}. {example}")
    
    # Insights
    print("\n💡 INSIGHT GENERATION")
    print("-" * 20)
    for category in examples.insight_examples():
        print(f"\n{category['category']}:")
        for i, example in enumerate(category['examples'][:3], 1):
            print(f"  {i}. {example}")
    
    # Advanced
    print("\n🔬 ADVANCED ANALYSIS")
    print("-" * 20)
    for category in examples.advanced_examples():
        print(f"\n{category['category']}:")
        for i, example in enumerate(category['examples'][:3], 1):
            print(f"  {i}. {example}")

def create_demo_notebook():
    """Create a Jupyter notebook with examples"""
    
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🤖 Agentic Data Analysis - Demo Notebook\n",
                    "\n",
                    "This notebook demonstrates how to use the Agentic Data Analysis system.\n",
                    "\n",
                    "## Setup\n",
                    "\n",
                    "First, make sure you have the required packages installed and your API key configured."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Import required libraries\n",
                    "import pandas as pd\n",
                    "import numpy as np\n", 
                    "from agent_nodes import DataAnalysisAgent\n",
                    "from langchain_core.messages import HumanMessage\n",
                    "\n",
                    "# Set up your API key\n",
                    "OPENAI_API_KEY = \"your-api-key-here\"  # Replace with your actual key\n",
                    "\n",
                    "# Initialize the agent\n",
                    "agent = DataAnalysisAgent(OPENAI_API_KEY)\n",
                    "workflow = agent.create_workflow()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Load sample data\n",
                    "df = pd.read_csv('sample_data.csv')\n",
                    "print(f\"Loaded dataset: {df.shape}\")\n",
                    "df.head()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Example 1: Basic data analysis\n",
                    "def ask_agent(question, dataframe):\n",
                    "    \"\"\"Helper function to interact with the agent\"\"\"\n",
                    "    config = {\"configurable\": {\"thread_id\": \"demo\"}}\n",
                    "    state = {\n",
                    "        \"messages\": [HumanMessage(content=question)],\n",
                    "        \"dataframe_json\": dataframe.to_json()\n",
                    "    }\n",
                    "    \n",
                    "    result = workflow.invoke(state, config)\n",
                    "    return result[\"messages\"][-1].content\n",
                    "\n",
                    "# Ask for data structure analysis\n",
                    "response = ask_agent(\"Analyze the structure of this dataset\", df)\n",
                    "print(response)"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    import json
    with open('demo_notebook.ipynb', 'w') as f:
        json.dump(notebook_content, f, indent=2)
    
    print("✅ Created demo notebook: demo_notebook.ipynb")

def run_interactive_demo():
    """Run an interactive demo session"""
    
    print("🎮 Interactive Demo Mode")
    print("Type 'quit' to exit, 'help' for examples")
    print("-" * 40)
    
    examples = AgentUsageExamples()
    
    while True:
        user_input = input("\n🤖 What would you like to analyze? ")
        
        if user_input.lower() == 'quit':
            print("👋 Thanks for trying the demo!")
            break
        elif user_input.lower() == 'help':
            print("\n💡 Try these example queries:")
            basic_examples = examples.basic_analysis_examples()[0]['examples']
            for i, example in enumerate(basic_examples[:5], 1):
                print(f"  {i}. {example}")
        else:
            print(f"\n🔄 Processing: '{user_input}'")
            print("📊 [In a real implementation, this would be sent to the agent]")
            print("✨ [The agent would analyze your data and provide insights]")
            print("📈 [Visualizations would be generated and displayed]")

if __name__ == "__main__":
    print("🚀 Agentic Data Analysis - Usage Examples")
    print("=" * 45)
    
    # Create sample dataset
    print("\n1. Creating sample dataset...")
    sample_df = create_sample_dataset("sample_data.csv")
    
    # Show dataset info
    print(f"\nSample dataset created with {len(sample_df)} rows:")
    print(sample_df.describe())
    
    # Demonstrate capabilities
    print("\n2. Demonstrating agent capabilities...")
    demonstrate_agent_capabilities()
    
    # Create demo notebook
    print("\n3. Creating demo resources...")
    create_demo_notebook()
    
    # Offer interactive demo
    print("\n" + "=" * 45)
    run_demo = input("Would you like to run the interactive demo? (y/n): ").lower()
    if run_demo == 'y':
        run_interactive_demo()
    
    print("\n🎉 Demo complete! Next steps:")
    print("1. Run: streamlit run streamlit_app_simplified.py")
    print("2. Upload the sample_data.csv file")
    print("3. Try the example queries shown above")
    print("4. Explore the demo notebook: demo_notebook.ipynb")