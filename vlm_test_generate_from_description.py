from openai import OpenAI

import base64
import requests

# Function to encode the image
# def encode_image(image_path):
#   with open(image_path, "rb") as image_file:
#     return base64.b64encode(image_file.read()).decode('utf-8')

#image_path1 = "/Users/ericsunkuan/Desktop/NTUEE/112-1/RL/final_project/vizdoom/ViZDoom-master/some_tests/game_image_save/game_image_sample2.jpg"
#base64_image1 = encode_image(image_path1)

client = OpenAI(api_key = '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')

response = client.chat.completions.create(
  model="gpt-3.5-turbo-1106",
  temperature=1.2,
  messages=[
    {
      "role": "user",
      
      "content": [
        {"type": "text", "text":"The below text enclosed in brackets { } is the description of the screen of a gameplay in the first person shooting game 'DOOM’, containing information of its location and the environment and the enemies.{This image showcases a scene from the game 'DOOM,' where the agent, seen in the first-person view, is within a gray-walled environment with a relatively dark ambiance. \n\n1. Walls and obstacles:\n   - Directly ahead, there is a continuous stone wall with an alternating block pattern, which extends beyond the right edge of the view.\n   - To the left about 45 degrees from the center of the agent's field of view is the corner of a wall, indicating a passage or opening to another area outside the immediate field of view. \n\n2. Enemy's position:\n   - There is an enemy at a location roughly to right 30 degrees from the center of the agent's field of view. It appears to be a 'Cacodemon,' a floating, monstrous creature associated with the 'DOOM' franchise. The enemy is located at a moderate distance, just off the right side of the wall that extends from the view.\n\n3. Item's position:\n   - No items resembling health packages or ammo are visible in the field of view shared in the image.\n\nUnder the limitations stated, this is all the information that can be drawn from the environment as presented in the picture.} Based on the specific situation described, please imagine you are a pro player and form a plan of 5 consecutive actions listed below to do at this moment so that you can optimize in navigating through the map and eliminating enemies. Please perceive that all enemies are able to deal long range attacks like shooting, so you need to consider the risk when collecting a health package in front of an enemy. Please output the number (an integer) and its direction (in degree format, let 90 degrees to the right be 0 and 90 degrees to the  left be 180) that represents the action only, and further explanation is not needed to be included in your output. For example, the output should be like : [1(30),3(60),2,4,6]. Below are the actions you can choose from : 1.Sprint in direction of 0, 30, 60, 90, 120, 150, 180 degrees 2.Aim and shoot  3.Take cover behind the wall or obstacle in direction of 0, 30, 60, 90, 120, 150, 180 degrees 4.Collect health package within sight 5.Collect ammo within sight 6.Explore the map"
},
        # Please tell me how many enemies are there in the agents view.
        # {
        #   # "type": "image_url",
        #   # "image_url": {
        #   #   "url": f"data:image/jpeg;base64,{base64_image1}",
        #   # },
        # },
      ],
    }
  ],
  max_tokens=1000,
)

print(response.choices[0].message.content)


