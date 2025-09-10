import re
import html
from modules import shared
from extensions.KokoroTTS_4_TGUI.src.debug import *

def clean_text_for_tts(text, preprocess_code=False):
    """
    Complete text cleaning for TTS - handles code blocks, markdown, and basic cleanup.
    Used by both standard and experimental modes.
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

    # Basic cleanup (always done)
    cleaned = cleaned.replace('*', '')  # Remove asterisks
    cleaned = cleaned.replace('`', '')  # Remove backticks (if not already handled)

    # Clean up extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    log(f"Cleaned text result: {cleaned[:100]}...")
    return cleaned

def handle_code_blocks(text):
    """
    Detect and replace code blocks with spoken announcements.
    Modify this function to customize how code is announced.
    """
    log("Processing code blocks")

    # Handle triple-backtick code blocks with language
    def replace_code_block_with_lang(match):
        language = match.group(1) if match.group(1) else "code"
        lines = match.group(2).strip().split('\n') if match.group(2) else []
        line_count = len([line for line in lines if line.strip()])

        # Customize these announcements as you like!
        if language.lower() in ['python', 'py']:
            return f" Python code block with {line_count} lines. "
        elif language.lower() in ['javascript', 'js']:
            return f" JavaScript code with {line_count} lines. "
        elif language.lower() in ['html']:
            return f" HTML code block. "
        elif language.lower() in ['css']:
            return f" CSS styling code. "
        elif language.lower() in ['bash', 'shell', 'sh']:
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
    Remove markdown formatting.
    """
    log("Cleaning markdown")

    # Remove headers
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)

    # Remove bold/italic
    text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)

    # Remove links [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    # Remove remaining HTML
    text = re.sub(r'<[^>]+>', '', text)

    return text

def simple_sentence_split(text, max_chunk_length=200):
    """
    Very simple sentence splitting for experimental mode.
    Just splits on periods, exclamation marks, and question marks.
    """
    if not text or not text.strip():
        return []

    log(f"Simple sentence splitting: {text[:50]}...")

    # Split on sentence endings
    sentences = re.split(r'([.!?]+\s*)', text)

    chunks = []
    current_chunk = ""

    for i in range(0, len(sentences), 2):
        sentence = sentences[i].strip() if i < len(sentences) else ""
        punctuation = sentences[i+1] if i+1 < len(sentences) else ""

        if sentence:
            full_sentence = sentence + punctuation

            # If current chunk would be too long, start new one
            if current_chunk and len(current_chunk + " " + full_sentence) > max_chunk_length:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = full_sentence
            else:
                current_chunk = current_chunk + " " + full_sentence if current_chunk else full_sentence

    # Add final chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Filter out very short chunks
    chunks = [chunk for chunk in chunks if len(chunk.strip()) > 3]

    log(f"Simple split: {len(chunks)} chunks")
    return chunks
