"""
Text preprocessing utilities for Tigrinya pretraining data.

Most rules in this module are corpus-driven: they were added after manually
inspecting raw text samples, tokenizer training logs, and artifacts surfaced
during the collection and cleaning of the datasets.
"""

import re
from collections import defaultdict
from hashlib import md5
from tqdm import tqdm
from data.constants import REMOVE_KEYWORDS


def apply_cleaning_rules(text):
    """
    Apply corpus-specific cleanup and quality rules to Tigrinya text.

    This is the final rule-based cleanup step in the preprocessing pipeline. It removes
    source artifacts such as BOM markers, page numbers, bullet prefixes, repeated
    punctuation, phone-number fragments, and dangling quotation marks. It also rejects
    segments that match known unwanted keywords, contain suspicious numeric patterns,
    have too many isolated single-letter tokens, or are too short to be useful.

    Args:
        text (str): A normalized Tigrinya text segment.

    Returns:
        Optional[str]: Cleaned text if the segment passes all rules, otherwise None.
    """
    # 1. Remove BOM character
    text = text.replace('﻿', '')

    # 2. Remove if contains any of the remove keywords
    if REMOVE_KEYWORDS.intersection(text.split()):
        return None

    # 3. Remove lines with page numbers (ቁ. NUMBER or ገጽ NUMBER)
    if re.search(r'ቁ\.\s*\d+|ገጽ\s*\d+', text):
        return None

    # 4. Replace three dots pattern (. . . or ...) with ''
    text = re.sub(r'\.\s*\.\s*\.|\.\.\.', '', text)

    # 5. Replace '=' with ''
    text = text.replace('=', '')

    # 6. Remove long sequences of numbers separated by spaces
    if re.search(r'\d+\s+\d+\s+\d+\s+\d+\s+\d+', text):
        return None

    # 7. Remove leading hyphen
    if text.strip().startswith('-'):
        text = text.lstrip('-').lstrip()

    # 8. Remove duplicate Tigrinya colon
    text = re.sub(r'፡\s+፡', '፡', text)

    # 9. Remove slashes with more than 2 repeated (/// or \\\)
    text = re.sub(r'[\\/]{3,}', '', text)

    # 10. Remove phone number pattern
    if re.search(r'ቍ\.ስ\s*\d+', text):
        return None

    # 10. Remove bullet points at start (•, ·, etc.)
    text = re.sub(r'^[•·▪▫‣⁃]\s*', '', text)

    # 11. Remove lines with single letters separated by spaces (more than 2)
    words = text.split()
    single_letter_count = sum(1 for w in words if len(w) == 1 and w.isalpha())
    if single_letter_count > 2:
        return None

    # 12. Remove zero-width and other problematic characters
    text = re.sub(r'[​-‍﻿]', '', text)

    # 13. Replace double quotes '' and '' with regular quotes "
    text = text.replace("''", '"')
    text = text.replace("''", '"')
    # Normalize smart quotes
    text = text.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")

    # Remove isolated quotes (like ' or " surrounded by spaces)
    text = re.sub(r"(^|\s)['\"](\s|$)", " ", text)

    # 14. Remove useless quotes at beginning/end
    text = text.strip()
    text = re.sub(r'^["\']+\s*', '', text)
    text = re.sub(r'\s*["\']+$', '', text)
    text = re.sub(r'^["\']\s+', '', text)
    text = re.sub(r'\s+["\']$', '', text)

    # 15. Discard very short text segments
    words = text.split()
    if len(words) <= 5:
        return None

    return text.strip()


def has_foreign_scripts(text):
    """
    Detect if the input text contains any character from scripts or alphabets
    that are not parts of the Tigrinya script.

    Characters from non-Tigrinya scripts observed in the raw corpus include:
      - Latin alphabet (English and extended Latin, covering transliteration letters like Ḥ, š, etc.)
      - CJK characters (Chinese, Japanese, Korean)
      - Greek
      - Cyrillic (Russian, etc.)
      - Hebrew
      - Arabic
      - Devanagari (Hindi and related scripts)

    This is used to filter out lines that may be contaminated with foreign characters.
    (Most surfaced in logs from tokenizer vocabulary analysis)

    Args:
        text (str): The input text string to check.

    Returns:
        bool: True if any foreign script character is present; False otherwise.
    """
    foreign_pattern = r'[a-zA-ZÀ-ɏḀ-ỿ一-鿿㐀-䶿぀-ゟ゠-ヿ가-힯Ͱ-ϿЀ-ӿ֐-׿؀-ۿऀ-ॿ-]'
    return bool(re.search(foreign_pattern, text))


def normalize_tigrinya_text(text):
    """
    Apply normalization to Tigrinya text by standardizing specific characters and punctuation.

    This function performs the following normalization steps:
        - Replaces all Tigrinya punctuation variants (፣, ፤, ፥, ፦, ᎓) with the canonical word separator '፡'.
        - Converts Western question marks ('?') to the Tigrinya question mark ('፧').
        - Converts colons (':') to the Tigrinya word separator ('፡').
        - Maps archaic or variant letter forms in the 'ሰ' and 'ጸ' families to their standard forms:
            * ሠ፡→፡ሰ, ሡ→ሱ, ሢ→ሲ, ሣ→ሳ, ሤ→ሴ, ሥ→ስ, ሦ→ሶ, ሧ→ሷ
            * ፀ→ጸ, ፁ→ጹ, ፂ→ጺ, ፃ→ጻ, ፄ→ጼ, ፅ→ጽ, ፆ→ጾ

    Args:
        text (str): Input Tigrinya string.

    Returns:
        str: The normalized text with canonical characters and punctuation.
    """
    # Normalize punctuation variants to '፡'
    # Range 639-642: ၊ ፤ ፥ ፦ -> ፡
    # Also ᎓ (᎓) -> ፡ (፡)
    text = re.sub(r'[፣፤፥፦᎓]', '፡', text)

    # Replace ? with ፧
    text = text.replace('?', '፧')

    # Replace : with ፡
    text = text.replace(':', '፡')

    # Normalize 'ሰ' family
    # ሠ -> ሰ, ሡ -> ሱ, ሢ -> ሲ, ሣ -> ሳ, ሤ -> ሴ, ሥ -> ስ, ሦ -> ሶ, ሧ->ሷ
    text = text.replace('ሠ', 'ሰ')
    text = text.replace('ሡ', 'ሱ')
    text = text.replace('ሢ', 'ሲ')
    text = text.replace('ሣ', 'ሳ')
    text = text.replace('ሤ', 'ሴ')
    text = text.replace('ሥ', 'ስ')
    text = text.replace('ሦ', 'ሶ')
    text = text.replace('ሧ', 'ሷ')

    # Normalize 'ጸ' family
    # ፀ -> ጸ, ፁ -> ጹ, ፂ -> ጺ, ፃ -> ጻ, ፄ -> ጼ, ፅ -> ጽ, ፆ -> ጾ
    text = text.replace('ፀ', 'ጸ')
    text = text.replace('ፁ', 'ጹ')
    text = text.replace('ፂ', 'ጺ')
    text = text.replace('ፃ', 'ጻ')
    text = text.replace('ፄ', 'ጼ')
    text = text.replace('ፅ', 'ጽ')
    text = text.replace('ፆ', 'ጾ')

    return text


def sanitize_tigrinya_text(text):
    """
    Remove a wide range of non-Tigrinya or unwanted symbols and characters from the input text.

    This function aggressively cleans up Tigrinya text by removing foreign, unwanted or
    extraneous symbols. 

    Cleaning steps performed:
      1. Remove an explicitly listed set of unwanted symbols
      2. Remove standard ASCII and Unicode brackets [ ], { }
      3. Remove a broad set of Unicode ranges that typically introduce foreign scripts or symbols
      4. Collapse whitespaces to a single space and trim the result.

    Args:
        text (str): Raw text

    Returns:
        str: The input text with all unwanted symbols removed.
    """
    # 1. Remove specific symbols identified
    chars_to_remove = r'[●◦❚《》√≤≥→−♡♢♤♧■□▲△▼▽◆◇○◎★☆†‡¢£¥€₹₤₯&"\\^`|~_]'
    text = re.sub(chars_to_remove, '', text)

    # 2. Remove brackets [] {}
    text = re.sub(r'[\[\]\{\}]', '', text)

    # 3. Remove unwanted unicode ranges
    # See docstring above for details.
    unwanted_ranges = (
        r'['
        r'-ɏ'  # Latin Ext
        r'ɐ-˿'  # IPA/Modifiers
        r'Ͱ-Ͽ'  # Greek
        r'Ѐ-ӿ'  # Cyrillic
        r'԰-֏'  # Armenian
        r'֐-׿'  # Hebrew
        r'؀-ۿ'  # Arabic
        r'ಀ-೿'  # Kannada
        r'က-႟'  # Myanmar
        r'፟'         # Gemination ፟
        r'፩-፼'  # Ethiopic Numbers
        r'፠፨'   # Ethiopic Symbols ፨ ፠
        r' -⁯'  # General Punctuation
        r'-'  # PUA
        r'�'         # Replacement Char
        r']'
    )
    text = re.sub(unwanted_ranges, '', text)

    # 4. Normalize spaces (collapse multiple spaces)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def preprocess_text(text):
    """
    Run the full text preprocessing pipeline on a Tigrinya text.

    Steps:
    1. Remove lines containing foreign scripts or characters.
    2. Apply comprehensive Tigrinya-specific cleaning rules.

    Args:
        text (str): The input text to filter,
            e.g.,  "ኣብ ገለ እዋን መዓት ዝጸሓፍ ኣብ ኣእምሮይ ይመጽእ እሞ ፡ ክጽሕፍ ኢለ ብርዕን ወረቐትን ምስ ሓዝኩ ህልም ይብለኒ ።"

    Returns:
        Optional[str]: The cleaned text if it passes all filtering steps,
        or None if the text is not valid.
    """
    if has_foreign_scripts(text):
        return None

    text = sanitize_tigrinya_text(text)
    text = normalize_tigrinya_text(text)
    cleaned = apply_cleaning_rules(text)

    return cleaned


def deduplicate_sentences_exact(sentence_list):
    """
    Deduplicate a list of sentences using MD5 hashing.

    Steps:
      1. Strip whitespace from each sentence.
      2. Compute MD5 hash for each stripped sentence.
      3. Keep only the first occurrence of each hash (i.e., unique sentence).
      4. Collect all subsequent sentences with the same hash as duplicates.

    Args:
        sentence_list (List[str]): List of sentences.

    Returns:
        Tuple containing:
          - unique_sentences (List[str]): Unique sentences (order preserved)
          - duplicate_sentences (Dict[str, List[str]]): Mapping from hash => list of duplicate sentences
          - unique_count (int): Number of unique sentences
          - duplicate_count (int): Number of duplicate sentences
    """
    seen_hashes = set()
    unique_count = 0
    duplicate_count = 0

    def get_hash(text):
        return md5(text.strip().encode('utf-8')).hexdigest()

    unique_sentences = []
    duplicate_sentences = defaultdict(list)

    for sentence in tqdm(sentence_list, desc="Deduplicating sentences"):
        sentence = sentence.strip()
        if not sentence:
            continue
        hash_value = get_hash(sentence)
        if hash_value not in seen_hashes:
            seen_hashes.add(hash_value)
            unique_count += 1
            unique_sentences.append(sentence)
        else:
            duplicate_count += 1
            duplicate_sentences[hash_value].append(sentence)

    return unique_sentences, duplicate_sentences, unique_count, duplicate_count
