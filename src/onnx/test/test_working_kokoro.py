# extract_working_vocab.py
from kokoro_onnx import Kokoro

kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

# Test the exact same text that's producing garbled output
test_text = "Hello, I'm Lily from the UK, how are you today?"

print(f"Testing: '{test_text}'")
print("=" * 60)

# Step-by-step comparison
phonemes = kokoro.tokenizer.phonemize(test_text)
tokens = kokoro.tokenizer.tokenize(test_text)

print(f"Working library phonemes: '{phonemes}'")
print(f"Working library tokens ({len(tokens)}): {tokens}")

# Generate audio with working library
import soundfile as sf
samples, sample_rate = kokoro.create(test_text, 'bf_lily')  # Same voice as your test
sf.write("working_kokoro_test.wav", samples, sample_rate)
print(f"Working library audio: {len(samples)} samples at {sample_rate}Hz")
print("Saved as working_kokoro_test.wav")

print("\n" + "=" * 60)
print("COMPARISON WITH YOUR IMPLEMENTATION:")
print("Your phonemes: 'həlˈoʊ, aɪm lˈɪli fɹʌmðə jˌuːkˈeɪ, hˌaʊ ɑːɹ juː tədˈeɪ?'")
print("Your tokens (55): [50, 83, 54, 132, 57, 135, 3, 16, 43, 102, 55, 16, 54, 132, 102, 54, 51, 16, 48, 123, 138, 55, 81, 83, 16, 52, 157, 63, 158, 53, 132, 47, 102, 3, 16, 50, 157, 43, 135, 16, 69, 158, 123, 16, 52, 63, 158, 16, 62, 83, 46, 132, 47, 102, 6]")

# Check if tokens match
your_tokens = [50, 83, 54, 132, 57, 135, 3, 16, 43, 102, 55, 16, 54, 132, 102, 54, 51, 16, 48, 123, 138, 55, 81, 83, 16, 52, 157, 63, 158, 53, 132, 47, 102, 3, 16, 50, 157, 43, 135, 16, 69, 158, 123, 16, 52, 63, 158, 16, 62, 83, 46, 132, 47, 102, 6]

if tokens == your_tokens:
    print("✅ TOKENS MATCH - Issue is in ONNX inference or voice processing")
else:
    print("❌ TOKENS DIFFER - Issue is in phonemization/tokenization")
    print(f"Length difference: working={len(tokens)}, yours={len(your_tokens)}")

# Test voice loading
print(f"\nVoice 'bf_lily' shape in working library: {kokoro.voices['bf_lily'].shape}")

# Extract vocabulary for comparison
working_vocab = kokoro.tokenizer.vocab
print(f"\nWorking vocabulary has {len(working_vocab)} entries")

# Save vocab to compare
with open("working_vocab.py", "w") as f:
    f.write("# Working vocabulary from kokoro-onnx library\n")
    f.write("WORKING_VOCAB = {\n")
    for symbol, token_id in sorted(working_vocab.items(), key=lambda x: x[1]):
        f.write(f"    '{symbol}': {token_id},\n")
    f.write("}\n")

print("Saved working vocabulary to working_vocab.py")
