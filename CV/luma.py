from lumaai import LumaAI
import os
import requests
import time

file = open(r"/Users/davidkim/security/lumaai.txt", "r", encoding='UTF8')
data = file.read()
KEY_LUMA = str(data)
file.close()

client = LumaAI(
    auth_token=KEY_LUMA,
)

# Basic
generation = client.generations.create(
    prompt="Make the alpaca looking left and right, simple green cosmic backgrounds, camera orbit right, middle speed.",
    loop=False,
    keyframes={
        # "frame0": {
        #     "type": "image",
        #     "url": "https://i.ibb.co/44bY42f/1.png"
        # },
        "frame1": {
            "type": "image",
            "url": "https://i.ibb.co/G3GJFvb/2.png"
        }
    }
)

# Extend
# generation = client.generations.create(
#     prompt="Make the alpaca looking left and right, simple green cosmic backgrounds, camera orbit right, middle speed.",
#     keyframes={
#       "frame0": {
#         "type": "generation",
#         "id": "0135ef49-5d8c-4e02-943e-902ec66f28a3"
#       }
#     }
# )

completed = False
print(f'### Dreaming loop started. ###')
while not completed:
    generation = client.generations.get(id=generation.id)
    if generation.state == "completed":
        completed = True
        print("### Completed. ###")
    elif generation.state == "failed":
        raise RuntimeError(f"Generation failed: {generation.failure_reason}")
        print("### Failed. ###")
        time.sleep(3)

video_url = generation.assets.video

response = requests.get(video_url, stream=True)
with open(f'/Users/davidkim/Desktop/{generation.id}.mp4', 'wb') as file:
    file.write(response.content)
print(f"### File downloaded as {generation.id}.mp4 ###")
