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
prompt = "The sky is"

messages = [
    {
        "role" : "user: ",
        "content" : prompt
    }
]

# call the generate_response()
response = generate_response(messages)
# Display the response
print(response)
