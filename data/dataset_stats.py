"""
Data statistics utilities for Tigrinya text datasets.
"""
import os
from collections import Counter
from typing import Optional

# Fonts that support Tigrinya script (for wordcloud)
TIGRINYA_FONTS = [
    # macOS
    '/System/Library/Fonts/Supplemental/Kefa.ttc',
    '/Library/Fonts/NotoSansEthiopic-Regular.ttf',
    # Linux
    '/usr/share/fonts/truetype/noto/NotoSansEthiopic-Regular.ttf',
    '/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-Regular.ttf',
    # Windows
    'C:/Windows/Fonts/nyala.ttf',
    'C:/Windows/Fonts/NotoSansEthiopic-Regular.ttf',
]


def get_tigrinya_font() -> Optional[str]:
    """
    Find the first available Tigrinya-compatible font on the host system.

    Returns:
        Path to a detected font file, or None if no known font exists.
    """
    for font_path in TIGRINYA_FONTS:
        if os.path.exists(font_path):
            return font_path
    return None

def get_file_size(filepath: str) -> str:
    """
    Read file size from disk and format it as a readable value.

    Args:
        filepath: Path to the file on disk.

    Returns:
        File size formatted as B/KB/MB/GB/TB.
    """
    size_bytes = os.path.getsize(filepath)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


def count_words(text: str) -> int:
    """
    Count whitespace-delimited words in a text blob.

    Args:
        text: Input text content.

    Returns:
        Total number of words.
    """
    return len(text.split())


def count_chars(text: str) -> int:
    """
    Count characters excluding spaces and newline characters.

    Args:
        text: Input text content.

    Returns:
        Total character count without spaces/newlines.
    """
    return len(text.replace(' ', '').replace('\n', ''))


def get_word_frequency(text: str, top_n: Optional[int] = None) -> Counter:
    """
    Get word frequency distribution.
    
    Args:
        text: Input text
        top_n: If provided, return only top N words
    
    Returns:
        Counter with word frequencies
    """
    words = text.split()
    freq = Counter(words)
    if top_n:
        return Counter(dict(freq.most_common(top_n)))
    return freq


def find_duplicates(text: str) -> dict:
    """
    Find duplicate lines and their counts.
    
    Args:
        text: Input text content.

    Returns:
        Summary dictionary with total/unique lines, duplicate metrics, and top duplicates.
    """
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    line_counts = Counter(lines)
    
    duplicates = {k: v for k, v in line_counts.items() if v > 1}
    top_duplicates = Counter(duplicates).most_common(10)
    
    total = len(lines)
    unique = len(line_counts)
    dup_count = total - unique
    
    return {
        'total_lines': total,
        'unique_lines': unique,
        'duplicate_count': dup_count,
        'duplicate_rate': dup_count / total * 100 if total > 0 else 0,
        'top_duplicates': top_duplicates
    }


def get_vocab_size(text: str) -> int:
    """
    Compute unique vocabulary size from whitespace tokenization.

    Args:
        text: Input text content.

    Returns:
        Number of unique tokens.
    """
    return len(set(text.split()))


def get_avg_line_length(text: str) -> float:
    """
    Compute average line length measured in words.

    Args:
        text: Input text content.

    Returns:
        Average number of words per non-empty line.
    """
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    if not lines:
        return 0
    return sum(len(line.split()) for line in lines) / len(lines)


def get_text_stats(text: str, filepath: Optional[str] = None) -> dict:
    """
    Get comprehensive text statistics.
    
    Args:
        text: Input text
        filepath: Optional file path for file size
    
    Returns:
        dict with all statistics
    """
    dup_info = find_duplicates(text)
    
    stats = {
        'file_size': get_file_size(filepath) if filepath else None,
        'total_lines': dup_info['total_lines'],
        'unique_lines': dup_info['unique_lines'],
        'duplicate_count': dup_info['duplicate_count'],
        'duplicate_rate': f"{dup_info['duplicate_rate']:.2f}%",
        'total_words': count_words(text),
        'total_chars': count_chars(text),
        'vocab_size': get_vocab_size(text),
        'avg_line_length': f"{get_avg_line_length(text):.1f} words",
    }
    return stats


def print_stats(stats: dict, title: str = "Dataset Statistics"):
    """
    Print statistics in a simple formatted table.

    Args:
        stats: Statistics dictionary to display.
        title: Table title shown at the top.

    Returns:
        None.
    """
    print(f"\n{'='*50}")
    print(f" {title}")
    print('='*50)
    for key, value in stats.items():
        if value is not None:
            label = key.replace('_', ' ').title()
            print(f"  {label:.<30} {value}")
    print('='*50)


def generate_wordcloud(text: str, width: int = 800, height: int = 400, 
                       max_words: int = 200, background_color: str = 'white',
                       font_path: Optional[str] = None):
    """
    Generate wordcloud from text.
    
    Args:
        text: Input text
        width: Image width
        height: Image height
        max_words: Maximum number of words
        background_color: Background color
        font_path: Path to font file (auto-detects a Tigrinya font if None)
    
    Returns:
        WordCloud object (requires wordcloud package)
    """
    try:
        from wordcloud import WordCloud
    except ImportError:
        raise ImportError("Please install wordcloud: pip install wordcloud")
    
    # Auto-detect Tigrinya font if not provided
    if font_path is None:
        font_path = get_tigrinya_font()
    
    wc = WordCloud(
        font_path=font_path,
        width=width,
        height=height,
        max_words=max_words,
        background_color=background_color,
        colormap='viridis',
        prefer_horizontal=0.7,
    )
    return wc.generate(text)

