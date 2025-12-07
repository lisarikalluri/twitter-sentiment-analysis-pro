"""Utils package for text preprocessing"""

try:
    from .preprocessing import TextPreprocessor
    __all__ = ['TextPreprocessor']
except ImportError:
    # If textblob or emoji not installed
    __all__ = []
    print("Warning: TextPreprocessor not available. Install textblob and emoji packages.")