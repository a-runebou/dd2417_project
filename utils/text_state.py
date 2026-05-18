import re

def split_context_prefix(text:str) -> tuple[str, str]:
    """
    Splits the typed text into context and prefix.
    Args:
        text: the typed text
        
    Returns:
        A tuple (context, prefix) where:
        - context: everything before the currently typed word
        - prefix: the currently typed pratial word
    """
    
    if text == "":
        return "", ""
    
    if text[-1].isspace():
        return text, ""
    
    match = re.search(r'(\S+)$', text)
    if match is None:
        return text, ""
    
    prefix = match.group(1)
    context = text[:match.start(1)]
    
    return context, prefix

