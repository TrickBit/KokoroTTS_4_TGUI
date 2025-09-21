"""
Text Processing Module for TTS

Handles text cleaning, code block processing, and markdown removal for TTS systems.
Provides configurable text preprocessing to convert complex text into speech-friendly format.
"""

def adjust_tgui_path():
    """
    Adjust sys.path to include the TGUI root directory (containing 'extensions/' or 'modules/').
    Call this before any imports to ensure consistent import paths in TGUI or standalone mode.
    """
    import os
    import sys
    start_dir = os.path.dirname(os.path.abspath(__file__))
    current = start_dir
    while current != os.path.dirname(current):  # Stop at filesystem root
        if os.path.exists(os.path.join(current, 'extensions')) and os.path.exists(os.path.join(current, 'modules')):
            if current not in sys.path:
                sys.path.insert(0, current)  # Prepend to prioritize
            return
        current = os.path.dirname(current)
    raise RuntimeError("Could not find TGUI root containing 'extensions/' or 'modules/'")

adjust_tgui_path()

import re
import html
from extensions.KokoroTTS_4_TGUI.src.debug import *


def clean_text_for_tts(text, preprocess_code=False):
    """
    Purpose: Complete text cleaning pipeline for TTS with optional code preprocessing
    Pre: Valid text string provided (may be empty)
    Post: Text cleaned and normalized for optimal TTS processing
    Args: text (str) - Input text to clean
          preprocess_code (bool) - Whether to process code blocks and markdown
    Returns: str - Cleaned text ready for TTS, empty string if input invalid
    """
    if not text or not text.strip():
        return ""

    log(f"Cleaning text for TTS: {text[:100]}...")

    # Basic HTML entity decoding
    cleaned = html.unescape(text)

    if preprocess_code:
        # Handle code blocks and markdown
        cleaned = handle_code_blocks(cleaned)
        cleaned = clean_markdown(cleaned)

    # Basic cleanup (always performed)
    cleaned = cleaned.replace('*', '')  # Remove asterisks
    cleaned = cleaned.replace('`', '')  # Remove backticks (if not already handled)

    # Clean up extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    log(f"Cleaned text result: {cleaned[:100]}...")
    return cleaned


def handle_code_blocks(text):
    """
    Purpose: Detect and replace code blocks with spoken announcements
    Pre: Valid text string provided
    Post: Code blocks converted to speech-friendly descriptions
    Args: text (str) - Text containing potential code blocks
    Returns: str - Text with code blocks replaced by spoken descriptions
    """
    log("Processing code blocks")

    def replace_code_block_with_lang(match):
        """
        Purpose: Convert code block with language specification to spoken form
        Pre: Valid regex match object containing language and code
        Post: Code block converted to descriptive announcement
        Args: match (Match) - Regex match containing language and code content
        Returns: str - Spoken description of the code block
        """
        language = match.group(1) if match.group(1) else "code"
        lines = match.group(2).strip().split('\n') if match.group(2) else []
        line_count = len([line for line in lines if line.strip()])

        # Customize announcements based on language
        language_lower = language.lower()
        if language_lower in ['python', 'py']:
            return f" Python code block with {line_count} lines. "
        elif language_lower in ['javascript', 'js']:
            return f" JavaScript code with {line_count} lines. "
        elif language_lower in ['html']:
            return f" HTML code block. "
        elif language_lower in ['css']:
            return f" CSS styling code. "
        elif language_lower in ['bash', 'shell', 'sh']:
            return f" Command line script. "
        else:
            return f" {language} code block. "

    # Replace ```language\ncode``` blocks
    text = re.sub(r'```(\w+)?\n(.*?)```', replace_code_block_with_lang, text, flags=re.DOTALL)

    # Handle inline code (`code`)
    text = re.sub(r'`([^`\n]{1,50})`', lambda m: f" code {m.group(1)} ", text)
    text = re.sub(r'`([^`\n]{51,})`', ' code snippet ', text)

    # Handle function calls and definitions without backticks
    text = re.sub(r'\b(def|function|class|import|from)\s+[\w.]+.*?:', ' Function definition. ', text)

    # Handle file extensions and paths
    text = re.sub(r'\b[\w/\\.-]*\.(py|js|html|css|json|yaml|txt|md|wav|mp3)\b', ' file ', text)

    # Handle URLs
    text = re.sub(r'https?://[^\s]+', ' web link ', text)

    return text


def clean_markdown(text):
    """
    Purpose: Remove markdown formatting while preserving readable content
    Pre: Valid text string with potential markdown formatting
    Post: Markdown syntax removed, plain text content preserved
    Args: text (str) - Text containing markdown formatting
    Returns: str - Plain text with markdown syntax removed
    """
    log("Cleaning markdown")

    # Remove headers (# ## ### etc.)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)

    # Remove bold/italic formatting (*text*, **text**, ***text***)
    text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)

    # Remove links [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    # Remove remaining HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    return text


def simple_sentence_split(text, max_chunk_length=200):
    """
    Purpose: Split text into sentences using simple punctuation-based rules
    Pre: Valid text string provided
    Post: Text split into sentence chunks, each under max_chunk_length
    Args: text (str) - Text to split into sentences
          max_chunk_length (int) - Maximum character length per chunk
    Returns: list - List of sentence strings, empty list if input invalid
    """
    if not text or not text.strip():
        return []

    log(f"Simple sentence splitting: {text[:50]}...")

    # Split on sentence endings (.!?)
    sentences = re.split(r'([.!?]+\s*)', text)

    chunks = []
    current_chunk = ""

    for i in range(0, len(sentences), 2):
        sentence = sentences[i].strip() if i < len(sentences) else ""
        punctuation = sentences[i+1] if i+1 < len(sentences) else ""

        if sentence:
            full_sentence = sentence + punctuation

            # If current chunk would exceed max length, start new chunk
            if current_chunk and len(current_chunk + " " + full_sentence) > max_chunk_length:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = full_sentence
            else:
                current_chunk = current_chunk + " " + full_sentence if current_chunk else full_sentence

    # Add final chunk if it contains content
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Filter out very short chunks (less than 4 characters)
    chunks = [chunk for chunk in chunks if len(chunk.strip()) > 3]

    log(f"Simple split: {len(chunks)} chunks")
    return chunks
