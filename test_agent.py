"""
Claude API Key Tester
Test your Anthropic API key before using the main application
"""

import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

def test_claude_connection(api_key, model="claude-3-haiku-20240307"):
    """Test Claude API connection"""
    try:
        llm = ChatAnthropic(
            model=model,
            temperature=0.1,
            api_key=api_key,
            max_tokens=100
        )
        
        response = llm.invoke([HumanMessage(content="Hello, can you respond with 'Claude API working perfectly'?")])
        return True, response.content
    except Exception as e:
        return False, str(e)

# Streamlit app
st.title("🧪 Claude API Key Tester")
st.markdown("Test your Anthropic API key before using the main data analysis app")

# Instructions
with st.expander("📋 How to get your Claude API Key"):
    st.markdown("""
    1. **Go to**: https://console.anthropic.com/
    2. **Sign up/Sign in** with your account
    3. **Navigate to**: Settings → API Keys
    4. **Create a new API key**
    5. **Copy the key** (starts with `sk-ant-`)
    6. **Paste it below** to test
    """)

# API Key input
api_key = st.text_input(
    "Enter your Anthropic API Key:", 
    type="password",
    placeholder="sk-ant-api03-..."
)

# Model selection
model = st.selectbox(
    "Select Claude Model:",
    [
        "claude-3-haiku-20240307",
        "claude-3-sonnet-20240229",
        "claude-3-opus-20240229"
    ]
)

# Test button
if api_key and st.button("🚀 Test Claude API Key", type="primary"):
    with st.spinner("Testing Claude connection..."):
        success, result = test_claude_connection(api_key, model)
        
        if success:
            st.success(f"✅ **Claude API Key works perfectly!**")
            st.info(f"**Claude's Response:** {result}")
            st.balloons()
        else:
            st.error(f"❌ **Claude API Key failed:** {result}")
            
            # Common troubleshooting tips
            with st.expander("🔧 Troubleshooting Tips"):
                st.markdown("""
                **Common issues:**
                1. **Invalid API Key**: Make sure you copied the full key starting with `sk-ant-`
                2. **No Credits**: Check your billing at https://console.anthropic.com/
                3. **Rate Limits**: Wait a moment and try again
                4. **Model Access**: Some models may require special access
                
                **Solutions:**
                - Create a new API key at https://console.anthropic.com/
                - Check your account status and billing
                - Try a different model (Haiku is usually most accessible)
                """)

# Info section
st.markdown("---")
st.markdown("### 📚 About Claude Models")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **🏃‍♂️ Claude 3 Haiku**
    - Fastest and most affordable
    - Great for data analysis
    - Recommended for testing
    """)

with col2:
    st.markdown("""
    **🎯 Claude 3 Sonnet**
    - Balanced speed and intelligence
    - Good for complex analysis
    - Most popular choice
    """)

with col3:
    st.markdown("""
    **🧠 Claude 3 Opus**
    - Most intelligent
    - Best for complex reasoning
    - Higher cost
    """)

st.info("💡 **Tip**: Start with Haiku for testing, then upgrade to Sonnet or Opus for production use!")