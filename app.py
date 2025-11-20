import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Configuration for your specific models
SUMMARIZER_MODEL_ID = "HussainR/t5-summarizer"
SENTIMENT_MODEL_ID = "HussainR/t5-sentiment-analysis"

class ModelManager:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Running on device: {self.device}")

    def load_model(self, model_id):
        """Lazy loader to load models only when needed to save RAM on startup."""
        if model_id not in self.models:
            try:
                print(f"Loading {model_id}...")
                self.tokenizers[model_id] = AutoTokenizer.from_pretrained(model_id)
                self.models[model_id] = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(self.device)
                print(f"Successfully loaded {model_id}")
            except Exception as e:
                return None, f"Error loading model: {str(e)}"
        return self.models[model_id], self.tokenizers[model_id]

    def summarize(self, text):
        if not text.strip():
            return "Please enter some text to summarize."
        
        model, tokenizer = self.load_model(SUMMARIZER_MODEL_ID)
        if isinstance(model, tuple): # Error caught
            return model[1]

        # T5 specific prefix for summarization
        input_text = "summarize: " + text
        
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        
        with torch.no_grad():
            summary_ids = model.generate(
                inputs.input_ids, 
                max_length=150, 
                min_length=40, 
                length_penalty=2.0, 
                num_beams=4, 
                early_stopping=True
            )
        
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def analyze_sentiment(self, text):
        if not text.strip():
            return "Please enter text to analyze."

        model, tokenizer = self.load_model(SENTIMENT_MODEL_ID)
        if isinstance(model, tuple): # Error caught
            return model[1]

        input_text = text 

        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs = model.generate(inputs.input_ids)
        
        sentiment = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "positive" in sentiment.lower():
            return f"😊 Positive ({sentiment})"
        elif "negative" in sentiment.lower():
            return f"😔 Negative ({sentiment})"
        else:
            return f"😐 {sentiment}"

    def process_both(self, text):
        """Helper to run both models on the same input."""
        summary = self.summarize(text)
        sentiment = self.analyze_sentiment(text)
        return summary, sentiment

# Initialize the manager
manager = ModelManager()

# --- Dark Mode Custom CSS ---
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

body {
    font-family: 'Poppins', sans-serif !important;
    background-color: #111827 !important; /* gray-900 */
    color: #f3f4f6 !important;
}

.gradio-container {
    max-width: 1100px !important;
}

/* Header Styling */
.header-container {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    color: white;
    padding: 3rem 2rem;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.4);
}

.header-title {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    letter-spacing: -0.025em;
}

.header-subtitle {
    font-size: 1.1rem;
    font-weight: 300;
    opacity: 0.9;
}

/* Card Styling for Columns - Dark Mode */
.content-card {
    background: #1f2937; /* gray-800 */
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.2);
    border: 1px solid #374151; /* gray-700 */
    height: 100%;
    color: white;
}

/* Fix for Markdown headers inside cards */
.content-card h3 {
    color: #f3f4f6 !important;
    margin-bottom: 1rem;
}

/* Force input text color to white (Replaces the failed python theme argument) */
textarea, input {
    color: white !important;
}

/* Button Styling */
.custom-btn {
    background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    border-radius: 10px !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    margin-top: 15px !important;
    font-size: 1.1rem !important;
}

.custom-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 20px -10px rgba(139, 92, 246, 0.5) !important;
}

/* Footer */
.footer-text {
    text-align: center;
    margin-top: 3rem;
    color: #9ca3af;
    font-size: 0.875rem;
}
"""

# --- UI Construction with Dark Theme ---
# We forcefully set the theme properties to dark colors
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="violet",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Poppins"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="#111827", # Dark background
    block_background_fill="#1f2937", # Darker gray for blocks
    block_border_width="0px",
    input_background_fill="#374151", # Input fields dark gray
    body_text_color_subdued="#9ca3af", 
    body_text_color="#f3f4f6",
    block_label_text_color="#e5e7eb"
    # Removed 'input_text_color' to fix the crash. Handled via CSS above.
)

with gr.Blocks(theme=theme, css=custom_css, title="NLP Dashboard") as demo:
    
    # 1. Beautiful Header
    gr.HTML("""
        <div class="header-container">
            <div class="header-title">NLP Intelligence Hub</div>
            <div class="header-subtitle">Unified Sentiment Analysis & Text Summarization</div>
        </div>
    """)
    
    with gr.Row():
        # LEFT COLUMN: Input
        with gr.Column(scale=1, elem_classes=["content-card"]):
            gr.Markdown("### 📝 Input Source")
            input_text = gr.Textbox(
                label="Text Content", 
                placeholder="Paste your article, review, or paragraph here to begin analysis...", 
                lines=12,
                show_label=False,
                container=False 
            )
            submit_btn = gr.Button("✨ Analyze Text", variant="primary", elem_classes=["custom-btn"])

        # RIGHT COLUMN: Outputs
        with gr.Column(scale=1, elem_classes=["content-card"]):
            gr.Markdown("### 📊 Analysis Results")
            
            # Sentiment Label
            output_sentiment = gr.Label(
                label="Detected Sentiment",
                num_top_classes=1,
                scale=0
            )
            
            # Summary Box
            gr.Markdown("#### Generated Summary")
            output_summary = gr.Textbox(
                label="Summary", 
                placeholder="Your summary will appear here...", 
                lines=8,
                interactive=False,
                show_copy_button=True,
                show_label=False,
                container=False,
                elem_id="summary-box"
            )

    # Example inputs
    gr.Examples(
        examples=[
            ["The product arrived late and was damaged. I am extremely disappointed with the service and will not be ordering again. It was a terrible experience."],
            ["I have tried several high-end headphones over the years, including Bose and Sony, but these are by far the most comfortable I have ever worn. The ear cups are soft and breathable, which is perfect for my long international flights. The noise cancellation is practically magic; it completely drowned out the engine drone on my last trip. The battery life claims 30 hours, and I actually got about 32 hours on a single charge. The only slight downside is that the app can be a bit finicky when trying to switch between devices, but once it connects, the connection is rock solid. Highly recommended for travelers!"],
            ["I wanted to love this coffee maker because it looks beautiful on my counter, but it has been nothing but a headache. First, the carafe leaks every single time you pour a cup, no matter how slowly you do it. It makes a huge mess on the table every morning. Second, the coffee never gets hot enough; it comes out lukewarm at best. After just two weeks of use, the digital display started glitching and now it won't even turn on. I contacted customer support three days ago and still haven't heard back. Save your money and buy a different brand."],
        ],
        inputs=input_text
    )

    # Logic connection
    submit_btn.click(
        fn=manager.process_both, 
        inputs=input_text, 
        outputs=[output_summary, output_sentiment]
    )

    # Footer
    gr.HTML("""
        <div class="footer-text">
            <p>Built using Gradio & Hugging Face Transformers | Models by HussainR</p>
        </div>
    """)

if __name__ == "__main__":
    demo.launch()