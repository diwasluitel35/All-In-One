import torch
from flask import Blueprint, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

news_summarizer_bp = Blueprint(
    'news_summarizer', 
    __name__,
    template_folder='templates',
    static_folder='static'
)

tokenizer = None
model = None
MODEL_PATH = 'diwasluitel/NewsSummarizer'

def get_model():

    global tokenizer, model
    if model is None:
        try:
            print(f"Loading News Summarizer from {MODEL_PATH}...")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
            print("News Summarizer loaded successfully!")
        except Exception as e:
            print(f"Critical Error loading model: {e}")
            return None, None
    return tokenizer, model

@news_summarizer_bp.route('/')
def home():

    return render_template('news/index.html')

@news_summarizer_bp.route('/summarize', methods=['POST'])
def summarize():

    tokenizer, model = get_model()

    data = request.get_json()
    news = data.get('news', '')

    if not news:
        return jsonify({'error': 'Please enter news to summarize.'}), 400

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        inputs = f"summarize: {news}"
        tokens = tokenizer(inputs, padding=False, max_length=1024, truncation=True, return_tensors='pt')
        tokens = {k: v.to(device) for k, v in tokens.items()}

        with torch.no_grad():
            outputs = model.generate(
                **tokens, 
                max_new_tokens=128, 
                min_new_tokens=20, 
                num_beams=4, 
                no_repeat_ngram_size=4, 
                early_stopping=False
            )

        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return jsonify({"summary": summary})
            
    except Exception as e:
        return jsonify({'error': f"Summarization failed: {str(e)}"}), 500