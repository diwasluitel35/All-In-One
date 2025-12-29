import torch
from flask import Blueprint, render_template, request, jsonify
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

essay_scorer_bp = Blueprint(
    'automated_essay_scorer', 
    __name__,
    template_folder='templates',
    static_folder='static'
)

tokenizer = None
model = None

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

class EssayScorer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model)
        hidden_size = self.transformer.config.hidden_size
        self.pooler = AttentionPooler(hidden_size)
        self.regressor = nn.Linear(hidden_size, 6)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = out.last_hidden_state
        pooled = self.pooler(last_hidden_state, attention_mask)
        scores = self.regressor(pooled)

        if labels is not None:
            loss = nn.MSELoss()(scores, labels)
            return {"loss": loss, "logits": scores}
        return {"logits": scores}

model_path = "diwasluitel/AutomatedEssayScorer"
base_model = 'microsoft/deberta-v3-base'

def get_model():

    global tokenizer, model
    if model is None:
        try:
            print(f"Loading Essay Scorer from {model_path}...")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = EssayScorer(model=base_model)
            model_weights_path = hf_hub_download(repo_id=model_path, filename="model.safetensors")
            state_dict = load_file(model_weights_path)
            model.load_state_dict(state_dict)
            print("Essay Scorer loaded successfully!")
        except Exception as e:
            print(f"Critical Error loading model: {e}")
            return None, None
    return tokenizer, model

@essay_scorer_bp.route('/')
def home():

    return render_template('essay/index.html')

@essay_scorer_bp.route('/predict', methods=['POST'])
def predict():

    tokenizer, model = get_model()

    data = request.get_json()
    essay = data.get('essay', '')

    if not essay:
        return jsonify({'error': 'Please enter an essay.'}), 400

    output_labels = ['Cohesion', 'Syntax', 'Vocabulary', 'Phraseology', 'Grammar', 'Conventions']
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        inputs = tokenizer(essay, return_tensors='pt', truncation=True, padding=False, max_length=1024)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        scores = outputs['logits'].cpu().numpy()[0]
        results = dict(zip(output_labels, scores))
            
    except Exception as e:
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 500

    formatted_results = {}
    for label, score in results.items():
        formatted_results[label] = f"{score:.1f}"

    return jsonify(formatted_results)