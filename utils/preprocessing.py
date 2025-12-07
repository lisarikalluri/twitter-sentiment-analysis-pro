import re
try:
    import emoji
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("Warning: textblob or emoji not installed")

class TextPreprocessor:
    def __init__(self):
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#(\w+)')
        
    def remove_urls(self, text):
        return self.url_pattern.sub('URL', text)
    
    def remove_mentions(self, text):
        return self.mention_pattern.sub('USER', text)
    
    def handle_hashtags(self, text):
        return self.hashtag_pattern.sub(r'\1', text)
    
    def handle_emojis(self, text):
        if not TEXTBLOB_AVAILABLE:
            return text
        try:
            return emoji.demojize(text, delimiters=(" ", " "))
        except:
            return text
    
    def remove_special_chars(self, text):
        text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
        return text
    
    def normalize_whitespace(self, text):
        return ' '.join(text.split())
    
    def preprocess(self, text, level='basic'):
        """
        Preprocess text with different levels:
        - basic: minimal cleaning (for deep learning models)
        - moderate: standard cleaning (for traditional ML)
        - aggressive: heavy cleaning (for simple models)
        """
        if not text or not isinstance(text, str):
            return ""
        
        text = text.lower()
        
        if level in ['moderate', 'aggressive']:
            text = self.remove_urls(text)
            text = self.remove_mentions(text)
        
        text = self.handle_hashtags(text)
        text = self.handle_emojis(text)
        
        if level == 'aggressive':
            text = self.remove_special_chars(text)
        
        text = self.normalize_whitespace(text)
        
        return text.strip()

def get_sentiment_polarity(text):
    """Get TextBlob sentiment polarity"""
    if not TEXTBLOB_AVAILABLE:
        return 0.0
    try:
        blob = TextBlob(text)
        return blob.sentiment.polarity
    except:
        return 0.0

def get_sentiment_subjectivity(text):
    """Get TextBlob sentiment subjectivity"""
    if not TEXTBLOB_AVAILABLE:
        return 0.0
    try:
        blob = TextBlob(text)
        return blob.sentiment.subjectivity
    except:
        return 0.0