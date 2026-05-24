from transformers import pipeline


def main():
    pipe = pipeline("text-generation", "gpt2")
    text = "who is worst president of USA in history."
    outputs = pipe(text, max_length=100, truncation=True, output_hidden_states=True)
    fixed_text = outputs[0]["generated_text"]
    print("--------------------------------")
    print("outputs is: ", fixed_text.removeprefix(text))


if __name__ == "__main__":
    main()
