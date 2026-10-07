import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "SicariusSicariiStuff/Impish_Mind_8B"
ADAPTER = "./impish-hindi-chat"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER)

print("Loading base model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.bfloat16,
    quantization_config=bnb_config,
    device_map="auto",
)

print("Loading Stage 2 adapter...")
model = PeftModel.from_pretrained(model, ADAPTER)

model.eval()

print("\nModel ready.")
print("Type 'exit' to quit.\n")

while True:
    prompt = input("You: ")

    if prompt.lower() == "exit":
        break

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    )

    inputs = {
        k: v.to(model.device)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    )

    print("AI:", response.strip())
    print()
