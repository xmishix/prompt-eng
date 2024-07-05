from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


model_id = "google/gemma-2b-it"                          # Identifier of pre-trained model
dtype = torch.bfloat16                                   # half-precision floating-point format that uses less memory

tokenizer = AutoTokenizer.from_pretrained(model_id)     # Loads tokenizer associated with the model. The tokenizer is responsible for converting input text into a format that the model can understand.
model = AutoModelForCausalLM.from_pretrained(           
    model_id, 
    device_map = "auto", 
    torch_dtype = dtype
) # Loads the actual model. It specifies the model ID, sets the device to CUDA (GPU), and defines the tensor data type.
    

# Function to generate the response
def generate_response(prompt):
    inputs = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    outputs = model.generate(inputs, max_new_tokens=150, num_return_sequences=1)
    response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    return response

# Store conversation history
conv_history = []

# Main chat loop
print("Mr. Chatbot is ready. Type 'quit' to exit. \nStart your conversation with Mr. Chatbot... Type a message: ")
while True:
    try:
        user_input = input("User: ")
        if user_input.lower() == "quit":
            break

        # Append user input to the conversation history
        conv_history.append({"role": "user", "content": user_input})    

        # Add the conversation history in the prompt
        prompt = tokenizer.apply_chat_template(conv_history, tokenize=False, add_generation_prompt=True) # This method returns the prepared prompt ready for text generation.
        # Prompt prepares the prompt for generating text. Tokenize = false indicates that the tokenizer should not tokenize the input. add_generation_prompt = True acutomatically prepend a generation prompt to the input text.

        # Generate a response
        bot_response = generate_response(prompt)
        print(f"Bot: {bot_response}")

        conv_history.append({"role": "assistant", "content": bot_response})


    except KeyboardInterrupt:
        print("\nExitng...")
        break

