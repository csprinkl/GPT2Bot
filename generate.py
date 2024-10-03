import model

def generate_response(prompt): #Take user input prompt and generate a GPT 2 bot response using the parameters defined.
    inputs = model.tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.model.generate(inputs, max_length=150, do_sample=True, top_p=0.95, top_k=60)
    response = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

conversation_history = [] #List to hold conversation history of the current bot interaction.
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    conversation_history.append(user_input)
    full_prompt = " ".join(conversation_history[-5:]) #Limit to last 5 exchanges for now.
    bot_response = generate_response(full_prompt)
    conversation_history.append(bot_response)
    print(f"Bot: {bot_response}")