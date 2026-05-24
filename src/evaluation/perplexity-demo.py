import evaluate
from transformers import pipeline
from utils import load_local_dataset, load_model_and_tokenizer


def format_to_text(example, tokenizer):
    messages = example.get("messages")
    if messages and tokenizer.chat_template:
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        return {"text": text}
    if "text" in example:
        return {"text": example["text"]}
    return {"text": str(example)}


model, tokenizer = load_model_and_tokenizer("SmolLM2-360M")
dataset = load_local_dataset("smoltalk2", config="Mid", split="train")

dataset = dataset.map(
    lambda x: format_to_text(x, tokenizer), remove_columns=dataset.column_names
)
texts = dataset["text"][:100]

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
evaluator = evaluate.load("perplexity")

result = evaluator.compute(model_or_pipeline=pipe, data=texts)

print(result)
