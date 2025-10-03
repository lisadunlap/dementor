import tiktoken

def get_token_count(text: str, model: str = "gpt-4") -> int:
    """
    Get the number of tokens in a text string using tiktoken.
    
    Args:
        text (str): The text to count tokens for
        model (str): The model to use for tokenization (default: gpt-4)
        
    Returns:
        int: Number of tokens in the text
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fall back to cl100k_base encoding if model not found
        encoding = tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(text))