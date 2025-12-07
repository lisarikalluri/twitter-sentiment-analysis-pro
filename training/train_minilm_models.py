import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

print("="*80)
print("🎓 TRAINING MINILM-BASED CLASSIFIERS")
print("="*80)

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

print(f"\n📁 Base Directory: {BASE_DIR}")
print(f"📁 Dataset Directory: {DATASET_DIR}")
print(f"📁 Model Directory: {MODEL_DIR}")

# Create directories
os.makedirs(MODEL_DIR, exist_ok=True)

# Load MiniLM model
print("\n📥 Loading MiniLM encoder...")
encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("   ✅ MiniLM loaded!")

# =============================================================================
# 1. SENTIMENT CLASSIFIER
# =============================================================================
print("\n" + "="*80)
print("1️⃣  TRAINING SENTIMENT CLASSIFIER")
print("="*80)

print("\n📊 Loading sentiment dataset...")
sentiment_file = os.path.join(DATASET_DIR, 'sentiment_train.csv')
df_sentiment = pd.read_csv(sentiment_file)

# Map text labels to numbers if needed
label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
if df_sentiment['sentiment'].dtype == 'object':
    df_sentiment['label'] = df_sentiment['sentiment'].map(label_map)
else:
    df_sentiment['label'] = df_sentiment['sentiment']

print(f"   📊 Dataset shape: {df_sentiment.shape}")
print(f"   📊 Label distribution:")
for label, name in enumerate(['negative', 'neutral', 'positive']):
    count = (df_sentiment['label'] == label).sum()
    print(f"      {label} ({name}): {count}")

print("\n🔄 Encoding texts with MiniLM...")
X_sentiment = encoder.encode(df_sentiment['text'].tolist(), show_progress_bar=True)
y_sentiment = df_sentiment['label'].values

print("\n🎯 Training Logistic Regression...")
sentiment_model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
sentiment_model.fit(X_sentiment, y_sentiment)

train_acc = accuracy_score(y_sentiment, sentiment_model.predict(X_sentiment))
print(f"   ✅ Training complete! Accuracy: {train_acc:.2%}")

# Save model
model_path = os.path.join(MODEL_DIR, 'minilm_sentiment.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(sentiment_model, f)
print(f"   💾 Saved to: {model_path}")

# =============================================================================
# 2. EMOTION CLASSIFIER
# =============================================================================
print("\n" + "="*80)
print("2️⃣  TRAINING EMOTION CLASSIFIER")
print("="*80)

print("\n📊 Loading emotion dataset...")
emotion_file = os.path.join(DATASET_DIR, 'emotion_train.csv')
df_emotion = pd.read_csv(emotion_file)

# Map emotion labels to numbers
emotion_map = {'anger': 0, 'fear': 1, 'joy': 2, 'sadness': 3, 'surprise': 4}
if df_emotion['emotion'].dtype == 'object':
    df_emotion['label'] = df_emotion['emotion'].map(emotion_map)
else:
    df_emotion['label'] = df_emotion['emotion']

print(f"   📊 Dataset shape: {df_emotion.shape}")
print(f"   📊 Label distribution:")
for label, name in enumerate(['anger', 'fear', 'joy', 'sadness', 'surprise']):
    count = (df_emotion['label'] == label).sum()
    print(f"      {label} ({name}): {count}")

print("\n🔄 Encoding texts with MiniLM...")
X_emotion = encoder.encode(df_emotion['text'].tolist(), show_progress_bar=True)
y_emotion = df_emotion['label'].values

print("\n🎯 Training Logistic Regression...")
emotion_model = LogisticRegression(max_iter=1000, random_state=42, C=1.0, multi_class='ovr')
emotion_model.fit(X_emotion, y_emotion)

train_acc = accuracy_score(y_emotion, emotion_model.predict(X_emotion))
print(f"   ✅ Training complete! Accuracy: {train_acc:.2%}")

# Save model
model_path = os.path.join(MODEL_DIR, 'minilm_emotion.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(emotion_model, f)
print(f"   💾 Saved to: {model_path}")

# =============================================================================
# 3. TOXICITY CLASSIFIER
# =============================================================================
print("\n" + "="*80)
print("3️⃣  TRAINING TOXICITY CLASSIFIER")
print("="*80)

print("\n📊 Loading toxicity dataset...")
toxicity_file = os.path.join(DATASET_DIR, 'toxicity_train.csv')
df_toxicity = pd.read_csv(toxicity_file)

# Create binary toxic label (1 if any toxicity column is 1)
if 'toxic' in df_toxicity.columns:
    toxic_cols = ['toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    df_toxicity['label'] = (df_toxicity[toxic_cols].sum(axis=1) > 0).astype(int)
else:
    df_toxicity['label'] = df_toxicity['label']

print(f"   📊 Dataset shape: {df_toxicity.shape}")
print(f"   📊 Label distribution:")
print(f"      0 (non-toxic): {(df_toxicity['label'] == 0).sum()}")
print(f"      1 (toxic):     {(df_toxicity['label'] == 1).sum()}")

print("\n🔄 Encoding texts with MiniLM...")
X_toxicity = encoder.encode(df_toxicity['text'].tolist(), show_progress_bar=True)
y_toxicity = df_toxicity['label'].values

print("\n🎯 Training Logistic Regression...")
toxicity_model = LogisticRegression(max_iter=1000, random_state=42, C=1.0, class_weight='balanced')
toxicity_model.fit(X_toxicity, y_toxicity)

train_acc = accuracy_score(y_toxicity, toxicity_model.predict(X_toxicity))
print(f"   ✅ Training complete! Accuracy: {train_acc:.2%}")

# Save model
model_path = os.path.join(MODEL_DIR, 'minilm_toxicity.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(toxicity_model, f)
print(f"   💾 Saved to: {model_path}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*80)
print("✅ ALL MINILM MODELS TRAINED SUCCESSFULLY!")
print("="*80)
print("\n📦 Saved Models:")
print(f"   1. Sentiment: {os.path.join(MODEL_DIR, 'minilm_sentiment.pkl')}")
print(f"   2. Emotion:   {os.path.join(MODEL_DIR, 'minilm_emotion.pkl')}")
print(f"   3. Toxicity:  {os.path.join(MODEL_DIR, 'minilm_toxicity.pkl')}")
print("\n🚀 Next: Train comparison models with train_comparison_models.py")
print("="*80)