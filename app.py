import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os
import sys

# Set up the page with better layout
st.set_page_config(
    page_title="Triple NLP System for Customer Feedback", 
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Main header - centered and clean
st.markdown(
    """
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-header">🧠 Triple NLP System for Customer Feedback</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Sentiment Analysis, Abstractive & Extractive Summarization</div>', unsafe_allow_html=True)

# Try to import her extractive summarizer from helper file
try:
    from extractive_helper import HybridSummarizer
    st.sidebar.success("✅ Extractive Summarization: Active (Team Code)")
except ImportError as e:
    st.sidebar.error("❌ Extractive Summarization: File missing")
    st.sidebar.error(f"Error: {e}")

# Check which models are available
st.sidebar.header("🔧 System Status")
if os.path.exists("final_sentiment_model"):
    st.sidebar.success("✅ Sentiment Analysis: Active")
else:
    st.sidebar.error("❌ Sentiment Analysis: Offline")

if os.path.exists("t5-summarizer_results"):
    st.sidebar.success("✅ Abstractive Summarization: Active")
else:
    st.sidebar.error("❌ Abstractive Summarization: Offline")

# Load ALL models
@st.cache_resource
def load_all_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    models = {}
    
    # Load trained sentiment model
    if os.path.exists("final_sentiment_model"):
        sentiment_tokenizer = AutoTokenizer.from_pretrained("final_sentiment_model")
        sentiment_model = AutoModelForSeq2SeqLM.from_pretrained("final_sentiment_model").to(device)
        models['sentiment'] = (sentiment_model, sentiment_tokenizer)
    
    # Load trained abstractive summarization model
    if os.path.exists("t5-summarizer_results"):
        abstractive_tokenizer = AutoTokenizer.from_pretrained("t5-summarizer_results")
        abstractive_model = AutoModelForSeq2SeqLM.from_pretrained("t5-summarizer_results").to(device)
        models['abstractive'] = (abstractive_model, abstractive_tokenizer)
    
    # Load extractive summarizer (HER CODE from helper file)
    models['extractive'] = HybridSummarizer()
    
    models['device'] = device
    return models

# Load models
with st.spinner('🔄 Initializing AI models...'):
    models = load_all_models()

# Prediction functions
def predict_sentiment(text):
    if 'sentiment' not in models:
        return "Model not loaded"
    
    model, tokenizer = models['sentiment']
    device = models['device']
    
    inputs = tokenizer(f"classify sentiment: {text}", return_tensors="pt", max_length=512, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=10, num_beams=1, early_stopping=True)
    
    sentiment = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return sentiment

def summarize_abstractive(text):
    if 'abstractive' not in models:
        return "Model not loaded"
    
    model, tokenizer = models['abstractive']
    device = models['device']
    
    inputs = tokenizer(f"summarize: {text}", return_tensors="pt", max_length=512, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        summary_ids = model.generate(
            **inputs,
            max_length=80,
            min_length=20,
            num_beams=2,
            do_sample=True,
            temperature=0.8,
            repetition_penalty=2.0,
            length_penalty=1.2,
            early_stopping=True
        )
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def summarize_extractive(text):
    return models['extractive'].summarize(text)

# Main input area with better styling
st.markdown("### 📝 Enter Customer Feedback")
user_input = st.text_area(
    "Paste or type customer feedback below:",
    height=150,
    placeholder="Example: I absolutely love this product! The quality is amazing, though the shipping took a bit longer than expected...",
    label_visibility="collapsed"
)

# Analyze button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_btn = st.button(
        "🚀 Analyze Feedback", 
        type="primary", 
        use_container_width=True,
        disabled=not user_input.strip()
    )

if analyze_btn and user_input.strip():
    
    with st.spinner('🤖 Analyzing with all models...'):
        # Use all three models
        sentiment = predict_sentiment(user_input)
        abstractive_summary = summarize_abstractive(user_input)
        extractive_summary = summarize_extractive(user_input)
        
        # Display results in clean cards
        st.markdown("---")
        st.markdown("### 📊 Analysis Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🎯 Sentiment Analysis")
            # Color code based on sentiment
            if "positive" in sentiment.lower():
                st.success(f"**Result:** {sentiment}")
                st.markdown("**Emotion:** 😊 Positive")
            elif "negative" in sentiment.lower():
                st.error(f"**Result:** {sentiment}")
                st.markdown("**Emotion:** 😠 Negative")
            else:
                st.warning(f"**Result:** {sentiment}")
                st.markdown("**Emotion:** 😐 Neutral")
            
        with col2:
            st.markdown("#### 📋 Abstractive Summary")
            st.info(abstractive_summary)
            st.caption("🤖 T5 Model - Generates new text")
            
        with col3:
            st.markdown("#### 📋 Extractive Summary")
            st.info(extractive_summary)
            st.caption("🔍 Hybrid Model - Selects key sentences")
        
        # Quick stats
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Input Length", f"{len(user_input.split())} words")
        with col2:
            st.metric("Abstractive Summary", f"{len(abstractive_summary.split())} words")
        with col3:
            st.metric("Extractive Summary", f"{len(extractive_summary.split())} words")
        with col4:
            compression = max(0, 100 - (len(abstractive_summary.split()) / len(user_input.split()) * 100))
            st.metric("Compression", f"{compression:.1f}%")

# Footer with info
st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 About")
st.sidebar.info(
    "This system analyzes customer feedback using three advanced NLP techniques: "
    "sentiment analysis, abstractive summarization (generates new text), and "
    "extractive summarization (selects key sentences)."
)

# Add some sample prompts
st.sidebar.markdown("### 🎯 Try These Examples")
sample_reviews = {
    "Positive Coffee Review": "I absolutely love this coffee maker! It brews perfectly every single time and the built-in grinder makes such a difference in flavor. The programmable timer is so convenient for my morning routine. Definitely worth every penny!",
    "Negative Restaurant Experience": "The food took over an hour to arrive and when it did, it was cold and poorly seasoned. The waiter was inattentive and never checked on us. The tables were dirty and the overall experience was terrible.",
    "Mixed Product Feedback": "The product itself is good quality and works as described, but the shipping took much longer than expected. The packaging was damaged when it arrived, though the item inside was fine. Customer service was responsive but couldn't speed up the delivery."
}

for name, review in sample_reviews.items():
    if st.sidebar.button(f"📄 {name}", key=name):
        st.session_state.user_input = review
        st.rerun()

# Model information
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Model Information")
st.sidebar.info("""
**Sentiment Analysis**: Fine-tuned FLAN-T5  
**Abstractive Summary**: Fine-tuned T5  
**Extractive Summary**: Hybrid TF-IDF + TextRank
""")