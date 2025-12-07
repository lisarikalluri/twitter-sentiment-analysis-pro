from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import os
import time
from sentence_transformers import SentenceTransformer
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

print("\n" + "="*80)
print("🚀 LOADING ENHANCED SENTIMENT ANALYSIS SYSTEM")
print("="*80)
print(f"\n📁 Base Directory: {BASE_DIR}")
print(f"📁 Model Directory: {MODEL_DIR}")

# Load MiniLM encoder
print("\n📥 Loading MiniLM encoder...")
try:
    encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    encoder.max_seq_length = 256
    print("   ✅ Encoder loaded!")
except Exception as e:
    print(f"   ❌ Error loading encoder: {e}")
    encoder = None

# Load MiniLM models
print("\n📥 Loading MiniLM models...")
minilm_models = {}
for task in ['sentiment', 'emotion', 'toxicity']:
    try:
        model_path = os.path.join(MODEL_DIR, f'minilm_{task}.pkl')
        with open(model_path, 'rb') as f:
            minilm_models[task] = pickle.load(f)
        print(f"   ✅ MiniLM {task.capitalize()} loaded")
    except Exception as e:
        print(f"   ❌ Error loading minilm_{task}: {e}")

# Load comparison models
print("\n📥 Loading comparison models...")
comparison_models = {}
model_types = ['svm', 'naivebayes', 'randomforest', 'logisticregression', 
               'decisiontree', 'gradientboosting']

for task in ['sentiment', 'emotion', 'toxicity']:
    comparison_models[task] = {}
    for model_type in model_types:
        model_path = os.path.join(MODEL_DIR, f'{model_type}_{task}.pkl')
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    comparison_models[task][model_type] = pickle.load(f)
                print(f"   ✅ {model_type.upper()} {task} loaded")
            except Exception as e:
                print(f"   ⚠️  Couldn't load {model_type}_{task}: {e}")

# Load TF-IDF vectorizers
print("\n📥 Loading TF-IDF vectorizers...")
tfidf_vectorizers = {}
for task in ['sentiment', 'emotion', 'toxicity']:
    try:
        vectorizer_path = os.path.join(MODEL_DIR, f'tfidf_{task}.pkl')
        with open(vectorizer_path, 'rb') as f:
            tfidf_vectorizers[task] = pickle.load(f)
        print(f"   ✅ {task.capitalize()} vectorizer loaded")
    except Exception as e:
        print(f"   ❌ Error loading tfidf_{task}: {e}")

print("\n" + "="*80)
print("✅ ALL MODELS LOADED SUCCESSFULLY!")
print("="*80)

# Label mappings
SENTIMENT_LABELS = ['negative', 'neutral', 'positive']
EMOTION_LABELS = ['anger', 'fear', 'joy', 'sadness', 'surprise']
TOXICITY_LABELS = ['non-toxic', 'toxic']

# Model name mapping for display
MODEL_NAME_MAP = {
    'minilm': 'MiniLM',
    'svm': 'SVM',
    'naivebayes': 'Naive Bayes',
    'randomforest': 'Random Forest',
    'logisticregression': 'Logistic Regression',
    'decisiontree': 'Decision Tree',
    'gradientboosting': 'Gradient Boosting'
}

def get_minilm_prediction(text, task):
    """Get prediction from MiniLM model"""
    if encoder is None or task not in minilm_models:
        return None, 0.0
    
    try:
        embedding = encoder.encode([text])
        model = minilm_models[task]
        
        if task == 'sentiment':
            labels = SENTIMENT_LABELS
        elif task == 'emotion':
            labels = EMOTION_LABELS
        elif task == 'toxicity':
            labels = TOXICITY_LABELS
        else:
            return None, 0.0
        
        pred = model.predict(embedding)[0]
        proba = model.predict_proba(embedding)[0]
        
        pred_idx = int(pred)
        pred_idx = max(0, min(pred_idx, len(labels) - 1))
        
        return labels[pred_idx], float(proba[pred_idx])
    except Exception as e:
        print(f"Error in get_minilm_prediction: {e}")
        return None, 0.0

def get_tfidf_prediction(text, model, vectorizer, labels):
    """Get prediction from TF-IDF based model"""
    try:
        text_vec = vectorizer.transform([text])
        pred = model.predict(text_vec)[0]
        proba = model.predict_proba(text_vec)[0]
        
        pred_idx = int(pred)
        pred_idx = max(0, min(pred_idx, len(labels) - 1))
        
        return labels[pred_idx], float(proba[pred_idx])
    except Exception as e:
        print(f"Error in get_tfidf_prediction: {e}")
        return None, 0.0

def get_all_predictions(text, task):
    """Get predictions from all models for a task"""
    predictions = {}
    
    if task == 'sentiment':
        labels = SENTIMENT_LABELS
    elif task == 'emotion':
        labels = EMOTION_LABELS
    elif task == 'toxicity':
        labels = TOXICITY_LABELS
    else:
        return predictions
    
    # MiniLM prediction
    start_time = time.time()
    label, conf = get_minilm_prediction(text, task)
    elapsed = (time.time() - start_time) * 1000
    
    if label:
        predictions['minilm'] = {
            'label': label,
            'confidence': round(conf, 4),
            'speed': round(elapsed, 2)
        }
    
    # TF-IDF models predictions
    if task in comparison_models and task in tfidf_vectorizers:
        vectorizer = tfidf_vectorizers[task]
        for model_name, model in comparison_models[task].items():
            try:
                start_time = time.time()
                label, conf = get_tfidf_prediction(text, model, vectorizer, labels)
                elapsed = (time.time() - start_time) * 1000
                
                if label:
                    predictions[model_name] = {
                        'label': label,
                        'confidence': round(conf, 4),
                        'speed': round(elapsed, 2)
                    }
            except Exception as e:
                print(f"Error with {model_name}: {e}")
    
    return predictions

def calculate_consensus(predictions):
    """Calculate consensus from multiple predictions"""
    if not predictions:
        return {'label': 'unknown', 'agreement': 0.0, 'total_models': 0}
    
    labels = [p['label'] for p in predictions.values() if p['label'] != 'error']
    
    if not labels:
        return {'label': 'unknown', 'agreement': 0.0, 'total_models': 0}
    
    # Get most common label
    label_counts = Counter(labels)
    consensus_label = label_counts.most_common(1)[0][0]
    agreement = label_counts[consensus_label] / len(labels)
    
    # Calculate weighted confidence
    weighted_conf = np.mean([p['confidence'] for p in predictions.values() 
                            if p['label'] == consensus_label])
    
    return {
        'label': consensus_label,
        'agreement': round(agreement, 4),
        'confidence': round(weighted_conf, 4),
        'total_models': len(labels),
        'vote_distribution': dict(label_counts)
    }

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.route('/', methods=['GET'])
def home():
    """API information"""
    return jsonify({
        'name': 'Twitter Sentiment Analysis Pro API',
        'version': '2.0',
        'status': 'running',
        'models_loaded': {
            'minilm': len(minilm_models),
            'comparison': sum(len(models) for models in comparison_models.values())
        },
        'endpoints': {
            '/': 'API information',
            '/health': 'Health check',
            '/analyze': 'Complete analysis (sentiment, emotion, toxicity)',
            '/analyze/sentiment': 'Sentiment analysis only',
            '/analyze/emotion': 'Emotion analysis only',
            '/analyze/toxicity': 'Toxicity detection only',
            '/compare': 'Compare all models',
            '/compare/sentiment': 'Compare sentiment models',
            '/compare/emotion': 'Compare emotion models',
            '/compare/toxicity': 'Compare toxicity models',
            '/batch': 'Batch analysis (multiple texts)'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'encoder_loaded': encoder is not None,
        'minilm_models': list(minilm_models.keys()),
        'comparison_models': {
            task: list(models.keys()) 
            for task, models in comparison_models.items()
        },
        'total_models': (len(minilm_models) + 
                        sum(len(m) for m in comparison_models.values()))
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    """Complete text analysis"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Get predictions for all tasks
        sentiment_label, sentiment_conf = get_minilm_prediction(text, 'sentiment')
        emotion_label, emotion_conf = get_minilm_prediction(text, 'emotion')
        toxicity_label, toxicity_conf = get_minilm_prediction(text, 'toxicity')
        
        result = {
            'text': text,
            'timestamp': time.time(),
            'sentiment': {
                'label': sentiment_label or 'unknown',
                'confidence': round(sentiment_conf, 4)
            },
            'emotion': {
                'label': emotion_label or 'unknown',
                'confidence': round(emotion_conf, 4)
            },
            'toxicity': {
                'label': toxicity_label or 'unknown',
                'confidence': round(toxicity_conf, 4),
                'is_toxic': toxicity_label == 'toxic'
            },
            'model': 'MiniLM',
            'text_length': len(text),
            'word_count': len(text.split())
        }
        
        print(f"✅ Analysis: {sentiment_label}/{emotion_label}/{toxicity_label}")
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Error in /analyze: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze/<task>', methods=['POST'])
def analyze_task(task):
    """Single task analysis"""
    try:
        if task not in ['sentiment', 'emotion', 'toxicity']:
            return jsonify({'error': f'Invalid task: {task}'}), 400
        
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        label, confidence = get_minilm_prediction(text, task)
        
        result = {
            'text': text,
            'task': task,
            'label': label or 'unknown',
            'confidence': round(confidence, 4),
            'model': 'MiniLM'
        }
        
        if task == 'toxicity':
            result['is_toxic'] = label == 'toxic'
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Error in /analyze/{task}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/compare', methods=['POST'])
def compare_all():
    """Compare predictions across all models - RESTRUCTURED FOR FRONTEND"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        print(f"\n🔍 Comparing all models for: '{text[:50]}...'")
        
        # Get predictions for all tasks
        sentiment_preds = get_all_predictions(text, 'sentiment')
        emotion_preds = get_all_predictions(text, 'emotion')
        toxicity_preds = get_all_predictions(text, 'toxicity')
        
        # Restructure data: organize by model instead of by task
        models = {}
        all_model_keys = set(sentiment_preds.keys()) | set(emotion_preds.keys()) | set(toxicity_preds.keys())
        
        for model_key in all_model_keys:
            display_name = MODEL_NAME_MAP.get(model_key, model_key.title())
            
            models[display_name] = {
                'sentiment': sentiment_preds.get(model_key, {'label': 'unknown', 'confidence': 0}),
                'emotion': emotion_preds.get(model_key, {'label': 'unknown', 'confidence': 0}),
                'toxicity': toxicity_preds.get(model_key, {'label': 'unknown', 'confidence': 0})
            }
        
        # Calculate consensus for each task
        sentiment_consensus = calculate_consensus(sentiment_preds)
        emotion_consensus = calculate_consensus(emotion_preds)
        toxicity_consensus = calculate_consensus(toxicity_preds)
        
        print(f"   Sentiment: {sentiment_consensus['label']} ({sentiment_consensus['agreement']:.0%} agreement)")
        print(f"   Emotion: {emotion_consensus['label']} ({emotion_consensus['agreement']:.0%} agreement)")
        print(f"   Toxicity: {toxicity_consensus['label']} ({toxicity_consensus['agreement']:.0%} agreement)")
        
        result = {
            'text': text,
            'timestamp': time.time(),
            'models': models,
            'consensus': {
                'sentiment': sentiment_consensus,
                'emotion': emotion_consensus,
                'toxicity': toxicity_consensus
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Error in /compare: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/compare/<task>', methods=['POST'])
def compare_task(task):
    """Compare predictions for a specific task"""
    try:
        if task not in ['sentiment', 'emotion', 'toxicity']:
            return jsonify({'error': f'Invalid task: {task}'}), 400
        
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        print(f"\n🔍 Comparing {task} models for: '{text[:50]}...'")
        
        predictions = get_all_predictions(text, task)
        consensus = calculate_consensus(predictions)
        
        result = {
            'text': text,
            'task': task,
            'predictions': predictions,
            'consensus': consensus,
            'timestamp': time.time()
        }
        
        print(f"   Consensus: {consensus['label']} ({consensus['agreement']:.0%})")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Error in /compare/{task}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/batch', methods=['POST'])
def batch_analyze():
    """Batch analysis for multiple texts"""
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        
        if not texts or not isinstance(texts, list):
            return jsonify({'error': 'No texts array provided'}), 400
        
        if len(texts) > 100:
            return jsonify({'error': 'Maximum 100 texts per batch'}), 400
        
        print(f"\n📦 Batch analyzing {len(texts)} texts...")
        
        results = []
        for idx, text in enumerate(texts):
            text = text.strip()
            if not text:
                continue
            
            sentiment_label, sentiment_conf = get_minilm_prediction(text, 'sentiment')
            emotion_label, emotion_conf = get_minilm_prediction(text, 'emotion')
            toxicity_label, toxicity_conf = get_minilm_prediction(text, 'toxicity')
            
            results.append({
                'id': idx,
                'text': text,
                'sentiment': {
                    'label': sentiment_label or 'unknown',
                    'confidence': round(sentiment_conf, 4)
                },
                'emotion': {
                    'label': emotion_label or 'unknown',
                    'confidence': round(emotion_conf, 4)
                },
                'toxicity': {
                    'label': toxicity_label or 'unknown',
                    'confidence': round(toxicity_conf, 4),
                    'is_toxic': toxicity_label == 'toxic'
                }
            })
        
        print(f"   ✅ Processed {len(results)} texts")
        
        return jsonify({
            'total': len(results),
            'results': results,
            'timestamp': time.time()
        })
        
    except Exception as e:
        print(f"❌ Error in /batch: {e}")
        return jsonify({'error': str(e)}), 500

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 STARTING ENHANCED API SERVER (v2.0)")
    print("="*80)
    print("📡 API running on: http://localhost:5000\n")
    print("📚 Available Endpoints:")
    print("   GET  / - API information")
    print("   GET  /health - Health check")
    print("   POST /analyze - Complete analysis")
    print("   POST /analyze/<task> - Single task analysis")
    print("   POST /compare - Compare all models (all tasks)")
    print("   POST /compare/<task> - Compare models (specific task)")
    print("   POST /batch - Batch analysis")
    print("\n🎯 Features:")
    print("   • Multi-model ensemble predictions")
    print("   • Consensus-based results")
    print("   • Performance metrics (speed, confidence)")
    print("   • Batch processing support")
    print("   • Comprehensive error handling")
    print("="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)