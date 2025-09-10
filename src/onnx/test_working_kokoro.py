# extract_working_vocab.py
from kokoro_onnx import Kokoro

kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

# Extract their complete vocabulary
working_vocab = kokoro.tokenizer.vocab
print("Complete working vocabulary:")
for symbol, token_id in sorted(working_vocab.items(), key=lambda x: x[1]):
    print(f"'{symbol}': {token_id}")

# Test tokenization step by step
text = "Hello"
phonemes = kokoro.tokenizer.phonemize(text)
tokens = kokoro.tokenizer.tokenize(text)

print(f"\nStep by step for '{text}':")
print(f"1. Phonemes: '{phonemes}'")
print(f"2. Tokens: {tokens}")

# Map each phoneme character to token
print(f"3. Character mapping:")
for i, char in enumerate(phonemes):
    if char in working_vocab:
        print(f"   '{char}' -> {working_vocab[char]}")
    else:
        print(f"   '{char}' -> NOT FOUND!")

import soundfile as sf
# Test basic generation
samples, sample_rate = kokoro.create(text,'af_bella')
# Save the audio
sf.write("working_test.wav", samples, sample_rate)
print(f"Generated audio: {len(samples)} samples at {sample_rate}Hz")
print("Saved as working_test.wav")
