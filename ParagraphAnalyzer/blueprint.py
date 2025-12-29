from flask import Blueprint, render_template, request, jsonify
import torch
import torch.nn as nn
import numpy as np
import joblib
from transformers import AutoTokenizer, AutoModel
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

paragraph_analyzer_bp = Blueprint(
    'paragraph_analyzer', 
    __name__,
    template_folder='templates',
    static_folder='static'
)

tokenizer = None
model = None
label_encoder = None

class AttentionPooler(nn.Module):
  def __init__(self, hidden_size):
    super().__init__()
    self.attention = nn.Linear(hidden_size, 1)

  def forward(self, last_hidden_state, attention_mask):
    attention_scores = self.attention(last_hidden_state)
    mask = attention_mask.unsqueeze(-1)
    attention_scores[mask == 0] = -1e4
    attention_weights = torch.softmax(attention_scores, dim=1)
    pooled_output = torch.sum(last_hidden_state * attention_weights, dim=1)
    return pooled_output

class ParagraphAnalyzer(nn.Module):
  def __init__(self, model):
    super().__init__()
    self.transformer = AutoModel.from_pretrained(model)
    hidden_size = self.transformer.config.hidden_size
    self.pooler = AttentionPooler(hidden_size)
    self.classifier = nn.Linear(hidden_size, 3)
    self.loss_fn = nn.CrossEntropyLoss()

  def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
    out = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
    last_hidden_state = out.last_hidden_state
    pooled = self.pooler(last_hidden_state, attention_mask)
    logits = self.classifier(pooled)

    if labels is not None:
        loss = self.loss_fn(logits, labels)
        return {"loss": loss, "logits": logits}
    return {"logits": logits}

model_path = 'diwasluitel/ParagraphAnalyzer'
base_model = 'microsoft/deberta-v3-base'

def get_model():

    global tokenizer, model, label_encoder
    if model is None:
        try:
            print(f"Loading Paragraph Analyzer from {model_path}...")
            path = hf_hub_download(repo_id=model_path, filename="label_encoder.pkl")
            label_encoder = joblib.load(path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = ParagraphAnalyzer(model=base_model)
            model_weights_path = hf_hub_download(repo_id=model_path, filename="model.safetensors")
            state_dict = load_file(model_weights_path)
            model.load_state_dict(state_dict, strict=False)
            print("Paragraph Analyzer loaded successfully!")
        except Exception as e:
            print(f"Critical Error loading model: {e}")
            return None, None, None
    return tokenizer, model, label_encoder

@paragraph_analyzer_bp.route('/')
def home():

    return render_template('paragraphA/index.html')

@paragraph_analyzer_bp.route('/analyze', methods=['POST'])
def analyze():

    tokenizer, model, label_encoder = get_model()

    data = request.get_json()
    text = data.get('text', '')
    type = data.get('type', '')

    if not text:
        return jsonify({'error': 'Please enter a paragraph.'}), 400

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        inputs = f"Discourse type: {type}. Text:{text}"
        tokens = tokenizer(inputs, padding=False, truncation=True, max_length=256, return_tensors='pt')
        tokens = {k: v.to(device) for k, v in tokens.items()}

        with torch.no_grad():
            outputs = model(**tokens)
            logits = outputs['logits']
            pred_id = np.argmax(logits.cpu().numpy(), axis=-1)
            pred_label = label_encoder.inverse_transform(pred_id)[0]

        return jsonify({"result": pred_label})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
