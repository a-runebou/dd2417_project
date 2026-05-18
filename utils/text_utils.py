from pathlib import Path


def read_text_files(paths: list[str]) -> str:
    """
    Read one+ text files and concatenate them 
    """
    texts = []
    
    for path in paths:
        p = Path(path)
        
        if not p.exists():
            raise FileNotFoundError(f"Could not find text file: {path}")
        
        texts.append(p.read_text(encoding="utf-8", errors="replace"))
    
    return "\n\n".join(texts)


def clean_text(text: str) -> str:
    """
    Normalize problematic characters found in some text files 
    """
    replacements = {
        "\x91": "'",
        "\x92": "'",
        "\x93": '"',
        "\x94": '"',
        "\x96": "-",
        "\x97": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-", 
        "\u2014": "-",
        "\xa0": " ",
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove control chars exect for newline and tab
    text = "".join(
        ch for ch in text
        if ch == "\n" or ch == "\t" or ord(ch) >= 32
    )
    return text


def read_and_clean_text_files(paths: list[str]) -> str:
    """
    Read and clean text files in one setp 
    """
    return clean_text(read_text_files(paths))
