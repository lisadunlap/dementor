import tiktoken

def get_token_count(text):
    """
    Returns a rough estimate of the number of tokens in a text.
    For more precise counting, especially with OpenAI models,
    use the tiktoken library.
    """
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    # Allow special tokens or disable the check for disallowed special tokens
    num_tokens = len(encoding.encode(text, disallowed_special=()))
    return num_tokens