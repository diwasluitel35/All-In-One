from flask import Flask, render_template
from AutomatedEssayScorer.blueprint import essay_scorer_bp
from NewsSummarizer.blueprint import news_summarizer_bp
from ParagraphAnalyzer.blueprint import paragraph_analyzer_bp
from ParagraphClassifier.blueprint import paragraph_classifier_bp

app = Flask(__name__, static_folder="static")

app.register_blueprint(essay_scorer_bp, url_prefix='/essay')
app.register_blueprint(news_summarizer_bp, url_prefix='/summarizer')
app.register_blueprint(paragraph_analyzer_bp, url_prefix='/analyzer')
app.register_blueprint(paragraph_classifier_bp, url_prefix='/classifier')

@app.route('/')
def home():
    return render_template('gateway.html')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)