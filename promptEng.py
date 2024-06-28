from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "google/gemma-2b-it"                                 # Identifier of pre-trained model
tokenizer = AutoTokenizer.from_pretrained(model_id)             # Loads tokenizer associated with the model. The tokenizer is responsible for converting input text into a format that the model can understand.
model = AutoModelForCausalLM.from_pretrained(model_id)          # Loads the actual model. 

# Function to generate the response
def generate_response(prompt):
    inputs = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    outputs = model.generate(inputs, max_new_tokens=150, num_return_sequences=1)
    response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    return response


# Basic example
prompt = """The odd numbers in this group add up to an even number: 4, 8, 9, 15, 12, 2, 1.
A: The answer is False.

The odd numbers in this group add up to an even number: 17,  10, 19, 4, 8, 12, 24.
A: The answer is True.

The odd numbers in this group add up to an even number: 16,  11, 14, 4, 8, 13, 24.
A: The answer is True.

The odd numbers in this group add up to an even number: 17,  9, 10, 12, 13, 4, 2.
A: The answer is False.

The odd numbers in this group add up to an even number: 15, 32, 5, 13, 82, 7, 1. 
A:"""

messages = [
    {
        "role": "user",
        "content": prompt
    }
]

# call the generate_response()
response = generate_response(messages)
# Display the response
print(response)
