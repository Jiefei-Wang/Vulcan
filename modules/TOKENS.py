class TOKENS:
    parent = "<|parent of|>"
    child = "<|child of|>"
    

TOKENS.all_tokens = [getattr(TOKENS, attr) for attr in dir(TOKENS) if not attr.startswith('__') and not callable(getattr(TOKENS, attr))]