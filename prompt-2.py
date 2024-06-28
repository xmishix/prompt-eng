prompt = """Context: Madrid is the capital of Spain.
Q: What is the capital of Spain?
A: Madrid is the capital of Spain.

Context: Dogs have four paws.
Q: How many paws does a dog have?
A: A dog has four paws.

Context: The sun is the center of our solar system.
Q: What is in the center of our solar system?
A: The sun is in the center of our solar system.

Context: The Earth is 4.543 billion years old.
Q: How old is the Earth?
A: 
"""

messages = [
    {
        "role" : "user: ",
        "content" : prompt
    }
]
