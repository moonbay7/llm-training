from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "SicariusSicariiStuff/Impish_Mind_8B"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype="auto",
    device_map="auto"
)

print("Model loaded successfully!")
print("Tokenizer vocabulary:", tokenizer.vocab_size)
