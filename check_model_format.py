#!/usr/bin/env python3
"""
Check the structure of saved models
"""

import pickle
import os

MODEL_DIR = 'C:/Users/LisariKalluri/Documents/Twitter_Sentiment_Analysis_Pro/models'

print("🔍 CHECKING MODEL FORMATS")
print("=" * 60)

for task in ['sentiment', 'emotion', 'toxicity']:
    model_path = os.path.join(MODEL_DIR, f'minilm_{task}.pkl')
    
    if os.path.exists(model_path):
        print(f"\n📦 Loading: minilm_{task}.pkl")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        print(f"   Type: {type(model)}")
        
        if isinstance(model, dict):
            print(f"   Structure: Dictionary with keys: {list(model.keys())}")
            for key, value in model.items():
                print(f"      - {key}: {type(value)}")
        else:
            print(f"   Structure: Direct model object")
            print(f"   Has predict_proba: {hasattr(model, 'predict_proba')}")
    else:
        print(f"\n❌ Not found: {model_path}")

print("\n" + "=" * 60)