import ollama

client = ollama.Client(host='http://127.0.0.1:11434')

response = client.chat(model='mistral', messages=[
    {'role': 'user', 'content': 'What is the capital of France?'},
])
print(response['message']['content'])