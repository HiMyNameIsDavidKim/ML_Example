import openai

file = open(r"/Users/davidkim/security/openai.txt", "r", encoding='UTF8')
data = file.read()
KEY_NLP = str(data)
file.close()

openai.api_key = KEY_NLP

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What can you do?"},
    ],
    temperature=1,
    n=4,
    # stop = [',', '.'],
    # max_tokens=1,
)

for res in response.choices:
    print(res.message)
