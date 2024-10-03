import model

def generate_response(prompt):
    inputs = model.tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.model.generate(inputs, max_length=150, do_sample=True, top_p=0.95, top_k=60)
    response = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    bot_response = generate_response(user_input)
    print(f"Bot: {bot_response}")