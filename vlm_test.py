from openai import OpenAI

import base64
import requests

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

image_path1 = "/Users/ericsunkuan/Desktop/NTUEE/112-1/RL/final_project/vizdoom/ViZDoom-master/some_tests/game_image_save/auto_map_buff.jpeg"

base64_image1 = encode_image(image_path1)

client = OpenAI(api_key = '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')

response = client.chat.completions.create(
  model="gpt-4-vision-preview",
  messages=[
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "This image is a map of the current level of the first person shooting game 'DOOM', the color of light brown represents the wall. Please memorize what the map looks like and I will input another image which is the screen that the agent sees."},
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image1}",
          },
        },
      ],
    }
  ],
  max_tokens=300,
)

print(response.choices[0])


image_path2 = "/Users/ericsunkuan/Desktop/NTUEE/112-1/RL/final_project/vizdoom/ViZDoom-master/some_tests/game_image_save/auto_map_buff_31.jpeg"

base64_image2 = encode_image(image_path2)

response2 = client.chat.completions.create(
  model="gpt-4-vision-preview",
  messages=[
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "This is the screen that the agents sees in the game I just mentioned, please combine the information in this image and the information you memorized of the map, then describe where the agent's location in the map."},
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image2}",
          },
        },
      ],
    }
  ],
  max_tokens=300,
)

print(response2.choices[0])

