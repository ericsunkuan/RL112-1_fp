from openai import OpenAI

import base64
import requests

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

image_path1 = "/Users/ericsunkuan/Desktop/Screenshots/截圖 2023-12-11 凌晨2.28.06.png"
base64_image1 = encode_image(image_path1)

client = OpenAI(api_key = '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')

response = client.chat.completions.create(
  model="gpt-4-vision-preview",
  temperature=1.2,
  messages=[
    {
      "role": "user",
      
      "content": [
        {"type": "text", "text": "This image is a screen of a gameplay in the first person shooting game 'DOOM’, containing information of its location and the environment and the enemies. Based on the specific situation shown in the image, please give me a detailed description of the environment the agent is seeing now, and please focus more on describing the wall’s and obstacle’s position and orientation, also please use the o’clock notation to specify the directions."
},
        # Please tell me how many enemies are there in the agents view.
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image1}",
          },
        },
      ],
    }
  ],
  max_tokens=1000,
)

print(response.choices[0])


