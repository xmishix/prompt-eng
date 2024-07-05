from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "google/gemma-2b-it"  # Replace with your model ID, e.g., "gpt2" or "t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

def CoT_response(prompt):
    CoT_response = f"{prompt}\n\nLet's think step by step." # Adding the Let's think step by step will prompt the model to think through the task and come up with an answer
    inputs = tokenizer.encode(CoT_response, return_tensors='pt').to(model.device)           # Encode the prompt and move it to the model's devide
    outputs = model.generate(inputs, max_new_tokens=500, num_return_sequences=1)                # Generate a response from the model
    response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)    # Decode the output and removes special tokens
    return response

# Prompt that will be use for Chain-of-Thought prompting for the Gemma model 
prompt = """Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?
A: Roger started with 5 balls. 2 cans of 3 balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11.

Q: The cafeteria has 25 apples originally. If they used 20 to make lunch and bought 6 more, how many apples do they have?

A: """

# prompt = input("Q: ")

# Call the CoT_response() function to create the response
response = CoT_response(prompt)
# Display the response
print(response)
