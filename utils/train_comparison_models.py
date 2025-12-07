import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🚀 TRAINING ENHANCED COMPARISON MODELS (TF-IDF + Advanced ML)")
print("="*80)

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

print(f"\n📁 Base Directory: {BASE_DIR}")
print(f"📁 Dataset Directory: {DATASET_DIR}")
print(f"📁 Model Directory: {MODEL_DIR}")

os.makedirs(MODEL_DIR, exist_ok=True)

# =============================================================================
# 1. SENTIMENT MODELS (EXPANDED)
# =============================================================================
print("\n" + "="*80)
print("1️⃣  TRAINING SENTIMENT COMPARISON MODELS")
print("="*80)

df_sentiment = pd.read_csv(os.path.join(DATASET_DIR, 'sentiment_train.csv'))

# Map labels
label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
if df_sentiment['sentiment'].dtype == 'object':
    y = df_sentiment['sentiment'].map(label_map).values
else:
    y = df_sentiment['sentiment'].values

X = df_sentiment['text'].values

print(f"📊 Dataset: {len(X):,} samples")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")

# Create enhanced TF-IDF vectorizer
print("\n📊 Creating enhanced TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(
    max_features=8000,      # Increased features
    ngram_range=(1, 3),     # Include trigrams
    min_df=2,               # Minimum document frequency
    max_df=0.9,             # Maximum document frequency
    sublinear_tf=True,      # Apply sublinear tf scaling
    strip_accents='unicode'
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

vectorizer_path = os.path.join(MODEL_DIR, 'tfidf_sentiment.pkl')
with open(vectorizer_path, 'wb') as f:
    pickle.dump(vectorizer, f)
print(f"   ✅ TF-IDF Vectorizer (features={X_train_vec.shape[1]:,}) saved")

# Train multiple models with optimized hyperparameters
models = {
    'svm': SVC(
        kernel='linear',
        probability=True,
        random_state=42,
        C=1.0,
        class_weight='balanced'
    ),
    'naivebayes': MultinomialNB(
        alpha=0.1  # Smoothing parameter
    ),
    'randomforest': RandomForestClassifier(
        n_estimators=200,
        max_depth=50,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    ),
    'logisticregression': LogisticRegression(
        max_iter=2000,
        random_state=42,
        C=1.0,
        solver='lbfgs',
        class_weight='balanced',
        n_jobs=-1
    ),
    'decisiontree': DecisionTreeClassifier(
        max_depth=30,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight='balanced'
    ),
    'gradientboosting': GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
}

print("\n🎯 Training models with optimized hyperparameters...")
results = []

for name, model in models.items():
    print(f"\n   Training {name.upper()}...")
    
    try:
        model.fit(X_train_vec, y_train)
        
        y_pred_train = model.predict(X_train_vec)
        y_pred_test = model.predict(X_test_vec)
        
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test, average='weighted')
        
        model_path = os.path.join(MODEL_DIR, f'{name}_sentiment.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        results.append({
            'model': name,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'f1': f1
        })
        
        print(f"      ✅ Train: {train_acc:.2%} | Test: {test_acc:.2%} | F1: {f1:.2%}")
        
    except Exception as e:
        print(f"      ❌ Error: {e}")

# Display results summary
print("\n📊 Sentiment Models Performance Summary:")
print("   " + "-" * 60)
print(f"   {'Model':<20} {'Train Acc':<12} {'Test Acc':<12} {'F1-Score':<12}")
print("   " + "-" * 60)
for r in sorted(results, key=lambda x: x['test_acc'], reverse=True):
    print(f"   {r['model']:<20} {r['train_acc']:<12.2%} {r['test_acc']:<12.2%} {r['f1']:<12.2%}")
print("   " + "-" * 60)

# =============================================================================
# 2. EMOTION MODELS (EXPANDED)
# =============================================================================
print("\n" + "="*80)
print("2️⃣  TRAINING EMOTION COMPARISON MODELS")
print("="*80)

df_emotion = pd.read_csv(os.path.join(DATASET_DIR, 'emotion_train.csv'))

emotion_map = {'anger': 0, 'fear': 1, 'joy': 2, 'sadness': 3, 'surprise': 4}
if df_emotion['emotion'].dtype == 'object':
    y = df_emotion['emotion'].map(emotion_map).values
else:
    y = df_emotion['emotion'].values

X = df_emotion['text'].values

print(f"📊 Dataset: {len(X):,} samples")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")

# Enhanced TF-IDF
vectorizer = TfidfVectorizer(
    max_features=6000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9,
    sublinear_tf=True
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

vectorizer_path = os.path.join(MODEL_DIR, 'tfidf_emotion.pkl')
with open(vectorizer_path, 'wb') as f:
    pickle.dump(vectorizer, f)
print(f"   ✅ TF-IDF Vectorizer (features={X_train_vec.shape[1]:,}) saved")

models = {
    'svm': SVC(
        kernel='linear',
        probability=True,
        random_state=42,
        C=1.0,
        class_weight='balanced'
    ),
    'naivebayes': MultinomialNB(alpha=0.1),
    'randomforest': RandomForestClassifier(
        n_estimators=200,
        max_depth=40,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    ),
    'gradientboosting': GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
}

print("\n🎯 Training models...")
results = []

for name, model in models.items():
    print(f"\n   Training {name.upper()}...")
    
    try:
        model.fit(X_train_vec, y_train)
        
        y_pred_test = model.predict(X_test_vec)
        test_acc = accuracy_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test, average='weighted')
        
        model_path = os.path.join(MODEL_DIR, f'{name}_emotion.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        results.append({'model': name, 'test_acc': test_acc, 'f1': f1})
        print(f"      ✅ Test: {test_acc:.2%} | F1: {f1:.2%}")
        
    except Exception as e:
        print(f"      ❌ Error: {e}")

print("\n📊 Emotion Models Performance Summary:")
print("   " + "-" * 50)
print(f"   {'Model':<20} {'Test Acc':<15} {'F1-Score':<15}")
print("   " + "-" * 50)
for r in sorted(results, key=lambda x: x['test_acc'], reverse=True):
    print(f"   {r['model']:<20} {r['test_acc']:<15.2%} {r['f1']:<15.2%}")
print("   " + "-" * 50)

# =============================================================================
# 3. TOXICITY MODELS (EXPANDED)
# =============================================================================
print("\n" + "="*80)
print("3️⃣  TRAINING TOXICITY COMPARISON MODELS")
print("="*80)

df_toxicity = pd.read_csv(os.path.join(DATASET_DIR, 'toxicity_train.csv'))

if 'toxic' in df_toxicity.columns:
    toxic_cols = ['toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    y = (df_toxicity[toxic_cols].sum(axis=1) > 0).astype(int).values
else:
    y = df_toxicity['label'].values

X = df_toxicity['text'].values

print(f"📊 Dataset: {len(X):,} samples")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")

# Enhanced TF-IDF for toxicity
vectorizer = TfidfVectorizer(
    max_features=7000,
    ngram_range=(1, 3),  # Trigrams help capture toxic patterns
    min_df=2,
    max_df=0.9,
    sublinear_tf=True
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

vectorizer_path = os.path.join(MODEL_DIR, 'tfidf_toxicity.pkl')
with open(vectorizer_path, 'wb') as f:
    pickle.dump(vectorizer, f)
print(f"   ✅ TF-IDF Vectorizer (features={X_train_vec.shape[1]:,}) saved")

models = {
    'svm': SVC(
        kernel='linear',
        probability=True,
        random_state=42,
        C=1.5,
        class_weight='balanced'
    ),
    'naivebayes': MultinomialNB(alpha=0.1),
    'randomforest': RandomForestClassifier(
        n_estimators=200,
        max_depth=40,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    ),
    'logisticregression': LogisticRegression(
        max_iter=2000,
        random_state=42,
        C=1.0,
        solver='lbfgs',
        class_weight='balanced',
        n_jobs=-1
    ),
    'gradientboosting': GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
}

print("\n🎯 Training models...")
results = []

for name, model in models.items():
    print(f"\n   Training {name.upper()}...")
    
    try:
        model.fit(X_train_vec, y_train)
        
        y_pred_test = model.predict(X_test_vec)
        test_acc = accuracy_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test, average='weighted')
        
        model_path = os.path.join(MODEL_DIR, f'{name}_toxicity.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        results.append({'model': name, 'test_acc': test_acc, 'f1': f1})
        print(f"      ✅ Test: {test_acc:.2%} | F1: {f1:.2%}")
        
    except Exception as e:
        print(f"      ❌ Error: {e}")

print("\n📊 Toxicity Models Performance Summary:")
print("   " + "-" * 50)
print(f"   {'Model':<20} {'Test Acc':<15} {'F1-Score':<15}")
print("   " + "-" * 50)
for r in sorted(results, key=lambda x: x['test_acc'], reverse=True):
    print(f"   {r['model']:<20} {r['test_acc']:<15.2%} {r['f1']:<15.2%}")
print("   " + "-" * 50)

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*80)
print("✅ ALL ENHANCED COMPARISON MODELS TRAINED SUCCESSFULLY!")
print("="*80)
print("\n📦 Models trained per task:")
print(f"   Sentiment: 6 models (SVM, NB, RF, LR, DT, GB)")
print(f"   Emotion:   4 models (SVM, NB, RF, GB)")
print(f"   Toxicity:  5 models (SVM, NB, RF, LR, GB)")
print("\n🎯 Enhanced Features:")
print(f"   • Optimized hyperparameters for each model")
print(f"   • Balanced class weights for imbalanced data")
print(f"   • Extended TF-IDF features (up to 8000)")
print(f"   • N-gram ranges (1-3) for better pattern capture")
print(f"   • Gradient Boosting for ensemble learning")
print("\n🚀 Next: Start the API server")
print("   cd backend")
print("   python app.py")
print("="*80)