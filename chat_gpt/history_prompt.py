### Prompting to directly generate an action set (14268) and the reasons it chose the set

"This image is the screen that my reinforcement learning agent sees in current level in the first person shooting game 'DOOM’, containing information of its location and the environment and the enemies. Based on the current situation shown in the image(including the enemies, where the walls and obstacles are located, health, etc.), please guess 3 possible actions that the agent can do at this moment so that the agent can eliminate the enemies in sight efficiently. Below are the actions you can choose from : Attack actions 1.Sprint in direction of 1 ranging to 12 o’clock direction (please give the exact number between 1 to 12) 2.Aim at closest enemy that is not blocked 3.Aim at closest enemy that is blocked 4.Fire Reposition actions  1.Take cover behind the wall or obstacle in direction of 1 ranging to 12 o’clock direction (please give the exact number between 1 to 12) 2.If no wall or obstacles in between maintain straight-line distance with closest enemy 3.Turn the camera view to direction of 1 ranging to 12 o’clock direction (please give the exact number between 1 to 12) Other actions  1.Collect health package within sight 2.Collect armor within sight 3.Collect ammo within sight"

"This image is the screen that my reinforcement learning agent sees in current level in the first person shooting game 'DOOM’, containing information of its location and the environment and the enemies. Based on the current situation shown in the image(including the enemies, where the walls and obstacles are located, health, etc.), please give advise by picking 3 possible actions that the agent can do at this moment so that the agent can navigate through the map and eliminate enemies, so that I can form reinforcement learning rewards base on your advise. Below are the actions you can choose from : Attack actions 1.Sprint in direction of 1 ranging to 12 o’clock direction (please give the exact number between 1 to 12) 2.Aim at closest enemy that is not blocked 3.Aim at closest enemy that is blocked 4.Fire Reposition actions  1.Take cover behind the wall or obstacle in direction of 1 ranging to 12 o’clock direction (please give the exact number between 1 to 12) 2.If no wall or obstacles in between maintain straight-line distance with closest enemy 3.Turn the camera view to direction of 1 ranging to 12 o’clock direction (please give the exact number between 1 to 12) Other actions  1.Collect health package within sight 2.Collect armor within sight 3.Collect ammo within sight"

"This image is a screen of a gameplay in the first person shooting game 'DOOM’, containing information of its location and the environment and the enemies. Based on the current situation shown in the image(including the enemies, where the walls and obstacles are located, health, etc.), please imagine 3 actions that an pro player will choose to do at this moment so that he can navigate through the map and eliminate enemies. Below are the actions you can choose from : Attack actions 1.Sprint in direction of 1 ranging to 12 o’clock direction (please give the exact number between 1 to 12) 2.Aim at closest enemy that is not blocked 3.Aim at closest enemy that is blocked 4.Fire Reposition actions  1.Take cover behind the wall or obstacle in direction of 1 ranging to 12 o’clock direction (please give the exact number between 1 to 12) 2.If no wall or obstacles in between maintain straight-line distance with closest enemy 3.Turn the camera view to direction of 1 ranging to 12 o’clock direction (please give the exact number between 1 to 12) Other actions  1.Collect health package within sight 2.Collect armor within sight 3.Collect ammo within sight"

"This image is a screen of a gameplay in the first person shooting game 'DOOM’, containing information of its location and the environment and the enemies. Based on the current situation shown in the image(including the enemies, where the walls and obstacles are located, health, etc.) and don't condider conditions that are not visible in the screen, please imagine you are a pro player and pick 3 actions listed below to do at this moment so that you can navigate through the map and eliminate enemies. Below are the actions you can choose from : Attack actions 1.Sprint in direction of 1 ranging to 12 o’clock direction (please give the exact number between 1 to 12) 2.Aim at closest enemy that is not blocked 3.Aim at closest enemy that is blocked 4.Fire Reposition actions  1.Take cover behind the wall or obstacle in direction of 1 ranging to 12 o’clock direction (please give the exact number between 1 to 12) 2.If no wall or obstacles in between maintain straight-line distance with closest enemy 3.Turn the camera view to direction of 1 ranging to 12 o’clock direction (please give the exact number between 1 to 12) Other actions  1.Collect health package within sight 2.Collect armor within sight 3.Collect ammo within sight"

"This image is the screen that my reinforcement learning agent sees in current level in the first person shooting game 'DOOM’, containing information of its environment and possibly some enemies. Based on the current situation shown in the image(including the enemies, where the walls and obstacles are located, health, etc.), please guess 3 possible actions that the agent can do at this moment so that the agent can eliminate the enemies in sight efficiently. Below are the actions you can choose from : Attack actions 1.Sprint in direction of 1 ranging to 12 o’clock direction (please give the exact number between 1 to 12) 2.Aim at closest enemy that is not blocked 3.Aim at closest enemy that is blocked 4.Fire Reposition actions  5.Take cover behind the wall or obstacle in direction of 1 ranging to 12 o’clock direction (please give the exact number between 1 to 12) 6.If no wall or obstacles in between maintain straight-line distance with closest enemy 7.Turn the camera view to direction of 1 ranging to 12 o’clock direction (please give the exact number between 1 to 12) Other actions  8.Collect health package within sight 9.Collect armor within sight 10.Collect ammo within sight"

"SUCCESS RATE >50 PROMPT :"
"This image is the screen that my reinforcement learning agent sees in current level in the first person shooting game 'DOOM’, containing information of its environment and possibly some enemies. Based on this specific situation shown in the image(including the enemies, where the walls and obstacles are located, health, etc.), please guess 3 possible actions that the agent can do at this moment so that the agent can eliminate the enemies in sight efficiently. Below are the actions you can choose from : Attack actions 1.Sprint in direction of 1 ranging to 12 o’clock direction (please give the exact number between 1 to 12) 2.Aim at closest enemy that is not blocked 3.Aim at closest enemy that is blocked 4.Fire Reposition actions  5.Take cover behind the wall or obstacle in direction of 1 ranging to 12 o’clock direction (please give the exact number between 1 to 12) 6.If no wall or obstacles in between maintain straight-line distance with closest enemy 7.Turn the camera view to direction of 1 ranging to 12 o’clock direction (please give the exact number between 1 to 12) Other actions  8.Collect health package within sight 9.Collect armor within sight 10.Collect ammo within sight"


"This image is the screen that my reinforcement learning agent sees in current level in the first person shooting game 'DOOM’, containing 1.information of its environment, 2.possibly some enemies or 3.collectable packages. Please recommend me a model that can extract the information in the image into a clean data structure. This will be used to train the reinforcement learning agent. Here’s an example of the data’s structure I want:\n\n```\nEnvironment:\n  Walls:\n    - Wall 1: Position (x1, y1), Length (l1), Orientation (vertical or horizontal)\n    - Wall 2: Position (x2, y2), Length (l2), Orientation (vertical or horizontal)\n    ...\n  \n  Obstacles (e.g., barrels, furniture, etc.):\n    - Obstacle 1: Type (e.g., barrel), Position (x1, y1)\n    - Obstacle 2: Type (e.g., candlestick), Position (x2, y2)\n    ...\n\nEnemies:\n  - Enemy 1: Type (e.g., soldier), Position (x1, y1), State (e.g., alert, patrolling)\n  - Enemy 2: Type (e.g., imp), Position (x2, y2), State (e.g., alert, patrol)\n  ...\n\nItems:\n  - Item 1: Type (e.g., health pack), Position (x1, y1), Quantity (q1)\n  - Item 2: Type (e.g., ammo), Position (x2, y2), Quantity (q2)\n  ...\n\nPlayer Status:\n  - Health: Current value\n  - Armor: Current value\n  - Ammo: Quantities for each weapon type\n  ...\n\nRadar/Sensor readings (if applicable):\n  - Directions (e.g., front, back, left, right)\n  - Detected Entities (e.g., enemies, obstacles)\n  ..."


### Prompting to generate a description of the environment 
# and focus on the walls,then generate action set with this prompt


### Part 1 (generate description)
{Prompt} success_rate < 40%
"This image is a screen of a gameplay in the first person shooting game 'DOOM’, containing information of its location and the environment and the enemies. Based on the specific situation shown in the image, please give me a detailed description of the environment the agent is seeing now, and please focus more on describing the wall’s position and orientation, also please use the o’clock notation to specify the directions."

{Output1}
image_path1 = "/Users/ericsunkuan/Desktop/NTUEE/112-1/RL/final_project/vizdoom/ViZDoom-master/some_tests/game_image_save/sample_in_game.png"
"In the image, the player appears to be in an outdoor nighttime environment characterized by a green floor with a texture that suggests some sort of energy or liquid effect, possibly radiation or slime, common themes in the 'DOOM' series. The sky is dark, implying it is either night or the area is in an enclosed space that mimics a night sky.\n\nFocusing on the walls and using o'clock notation:\n\n- Directly ahead and center (12 o'clock), there is a tall gray stone wall with a fairly uniform look resembling blocks or bricks. In the distance, there appears to be a monstrously large creature with glowing red eyes, indicative of a probable enemy at about 11 to 12 o'clock.\n\n- To the right of that enemy creature, still at 12 o'clock but slightly farther to the right side, is a clear path between two segments of the wall. Behind this gap at a vague 1 o'clock direction (further in the background), another similar enemy can be discerned from afar.\n\n- At about 2-3 o'clock is another area of walls, composed of the same gray stone, that form a corner leading off to the right edge of the screen. The walls in that direction are about two character heights tall.\n\n- Prominently, at approximately 4-5 o'clock (closer to the player's perspective), there is an isolated section of the same gray wall style physically closer to the player than the rest, implying that it extends outward, closer to the center of the area the player is in.\n\n- Rotating the view around in a hypothetical sense, towards 6 o'clock, there would be a view directly back from the initial viewpoint, but this is not visible in the image.\n\n- To the player's immediate left, resembling a 9 o'clock orientation, we cannot see if there's any wall or not due to obstruction from the gun/mechanism.\n\n- Additional elements visible include a health pack directly in front of the player (around 11 o'clock) and an ammunition object also in view further to the right roughly around 2 o'clock.\n\nThis environment suggests a combat situation where the player may need to navigate or fight, keeping spatial positioning in mind for strategic maneuvering."
'The environment shown in the image appears to be an outdoor area with a dark sky overhead, perhaps simulating nighttime or a space with low lighting. The agent is located in what seems to be a courtyard with green flooring that has an animated, wavy texture providing an illusion of movement, suggesting possibly some hazardous or toxic ground.\n\nDirectly ahead, at the approximate 12 o\'clock position, there is a large, elevated enemy with red eyes and a brownish body, positioned slightly to the left of center from the agent\'s perspective. To the right of this enemy, at around 1 o\'clock, another enemy is partially obscured by a corner of a wall.\n\nThere is a gray wall directly in front of the agent, slightly to the left extending perpendicularly relative to the agent\'s perspective, starting at approximately 11 o\'clock and continuing out of view to the left. A right-angle corner is visible where two walls meet, though the perpendicular wall blocks visibility beyond this point to the left.\n\nThe corner of another wall is evident to the right, proceeding from approximately 2 o\'clock and receding towards the right out of the player\'s immediate view. This wall forms an "L" shape with another segment of a wall that is parallel to the agent\'s line of sight and begins at the previously described wall corner around 2 o\'clock. The space between the left and right structures at the center of view appears to be a pathway, possibly leading to another area or serving as a tactical environment in the game for the agent to navigate.\n\nCloser to the player\'s immediate position, an item with a red cross, which commonly signifies a health pickup in many video games, sits squarely at around 12 o\'clock on the ground, in between the player and the central enemy. Additionally, there are smaller pickups, including at least one ammunition casing that can be seen slightly to the right on the ground at around 1:30.\n\nLastly, we can see the player\'s hand towards the bottom center of the screen, gripping what seems to be a pistol—a common weapon used in first-person shooters. This provides context regarding the game\'s interactivity and the possible actions available to the player, such as moving and shooting.'


{Output2}
image_path1 = "/Users/ericsunkuan/Desktop/NTUEE/112-1/RL/final_project/vizdoom/ViZDoom-master/some_tests/game_image_save/game_image_sample3.jpeg"
"This image depicts a three-dimensional first-person scene from the classic video game DOOM. The player is in an enclosed, dimly lit, corridor-like environment with textured brown walls and features two fellow marines or enemies facing the player within close proximity.\n\nStraight ahead, at 12 o'clock, a wall is visible directly in front of the player with an apparent exit or entryway to the player's right at around 1-2 o'clock, which is obscured partially by one of the characters in the foreground. The walls to the immediate left and right (9 o'clock and 3 o'clock respectively) are parallel to the player's view and extend towards what can be inferred as the back of the room, converging at a corner which is positioned approximately at 1:30 o'clock.\n\nThe two humanoid figures occupy the space in the middle of the corridor, with one situated almost directly at 12 o'clock position and the other slightly to the right, around 1 o'clock. Their orientation suggests they might be facing each other in play.\n\nFrom the perspective provided, it's not possible to determine the full layout beyond what is immediately visible, nor is it possible to see if there are any additional paths branching from behind the player or just out of sight.\n\nPlease note that the depiction of these environments and the characters within them is based on pixelated, stylized graphics characteristic of video games from the early '90s and lacks real-world environmental detail and accuracy."

{Output3}
image_path1 = "/Users/ericsunkuan/Desktop/NTUEE/112-1/RL/final_project/vizdoom/ViZDoom-master/some_tests/game_image_save/no_enemy_sample.jpeg"
"The image displays a scene from the first-person shooter game DOOM with retro-styled graphics. The environment appears to be an indoor area with a predominantly industrial or military design, consistent with the DOOM aesthetic.\n\nDirectly in the 12 o'clock position, there is an opening that consists of a lighter-colored wall frame, which could be a doorway to another room or corridor—this opening stretches from approximately the 11 to the 1 o'clock positions. The walls adjacent to this opening, at roughly the 10:30 and 1:30 o'clock positions, recede away from the player's perspective and feature a vertical striped texture, suggesting a constructed barrier or partition.\n\nThese side walls seem to border a walkway or platform that extends across the image's foreground, leading toward the opening at 12 o'clock. Below this walkway, presumably a step lower, is a blue floor or possibly water that occupies the vertical field from approximately the 6 to 9 o'clock positions on the left and from 6 to 3 o'clock on the right, filling the bottom half of the visual frame.\n\nThe floor area directly in the center of the image, extending from the blue area at the bottom toward the center (from 6 to about the 8:30 position), appears to be carpeted or tiled in a different pattern and elevation from the surroundings, implicating a pathway or notable section of the floor. The environment above the player's view, from approximately 11 to 1 o'clock positions, shows a ceiling that is decorated with hanging light fixtures or possibly skylights.\n\nAs for enemies or items, the image presents what appears to be a slain enemy lying on the ground at the center of the pathway (around 12 o'clock), a considerable distance away from the player's first-person viewpoint.\n\nThe environment depicted has a strong symmetrical design, with the pathway central to the room and walls extending away on either side. The inferred movement could be either towards the doorway at 12 o'clock or laterally along the platforms edging the higher walkway level. Please note that due to the low-resolution, pixelated style of the original DOOM game, certain environmental details might not be clear and might require some interpretation."

{Output4}
image_path1 = "/Users/ericsunkuan/Desktop/NTUEE/112-1/RL/final_project/vizdoom/ViZDoom-master/some_tests/game_image_save/game_image_sample2.jpg"
"In the image, the agent is standing in an outdoor environment that portrays a dystopian and desolate atmosphere typical of the game 'DOOM.' Let us focus on the surroundings as per the o'clock directions, from the perspective of the player:\n\nStraight ahead (12 o'clock): There is an enemy creature, which appears to be a humanoid demon, in the middle ground. Behind it, a bit further away, is another enemy bearing a firearm, standing in front of a gray stone wall with structured patterns, resembling a fortress wall. They are both centered in the view and appear to be facing the player.\n\nTo the immediate right (between 1 and 3 o'clock): The environment extends into an open area without any obstacles right next to the agent, and there's open space leading to the distant walls.\n\nTo the left (between 9 and 11 o'clock): There is a raised green ledge or plateau. Above this ledge, the stone wall, similar to the one straight ahead, is visible and it seems to form a continuous barrier surrounding the area. The plateau is not immediately next to the agent, allowing some room for maneuvering between the agent's current position and the raised ground.\n\nAhead to the left (between 11 and 12 o'clock): In the middle distance is the corner of a structure, possibly the beginning of another wall, forming a right angle with the wall ahead and possibly connecting with the wall seen to the left above the green plateau.\n\nThe surrounding ground appears to be a rough, uneven natural terrain, primarily in tones of brown and green, suggestive of dirt and grass or moss. Small red objects, which might be either items or decorative elements, dot the terrain.\n\nVarious status indicators are overlaid on the image, including the player's health, ammunition, arms, and armor details, but they don't represent physical elements within the game's environment. This heads-up display (HUD) is factored into the game's presentation to give the player situational awareness."

### Part 2 (generate actions)

{Prompt}
"The below text enclosed in brackets {} is the description of the screen of a gameplay in the first person shooting game 'DOOM’, containing information of its location and the environment and the enemies.{} Based on the specific situation described, please imagine you are a pro player and pick 3 actions listed below to do at this moment so that you can navigate through the map and eliminate enemies. Below are the actions you can choose from : Attack actions 1.Sprint in direction of 1 ranging to 12 o’clock direction (please give the exact number between 1 to 12) 2.Aim at closest enemy that is not blocked 3.Aim at closest enemy that is blocked 4.Fire Reposition actions  1.Take cover behind the wall or obstacle in direction of 1 ranging to 12 o’clock direction (please give the exact number between 1 to 12) 2.If no wall or obstacles in between maintain straight-line distance with closest enemy 3.Turn the camera view to direction of 1 ranging to 12 o’clock direction (please give the exact number between 1 to 12) Other actions  1.Collect health package within sight 2.Collect armor within sight 3.Collect ammo within sight"
{The description generated from Part1 goes into the brackets}

{Output1}
"As a pro player analyzing this situation in 'DOOM', and considering the layout described, I would carry out the following actions to optimally navigate the map and engage enemies:\n\n1. Take cover behind the wall or obstacle at 5 o’clock:\nGiven that this wall section is prominent and closer to the player's perspective, it seems like a tactical choice for taking immediate cover. This proximity allows for a defensive position to be held while giving me a chance to assess the battlefield or reload if necessary.\n\n2. Aim at closest enemy that is not blocked:\nOnce in cover and with an understanding of my immediate surroundings, targeting the closest visible enemy which is at about 11 to 12 o'clock is the next move. Aiming and being ready to fire at this enemy reduces the threat quickly before they can inflict too much damage.\n\n3. Collect health package within sight:\nAssuming that my health might be less than optimal, the health pack at 11 o'clock is a good acquisition. Care would be taken to sprint to it and quickly retreat to cover to minimize vulnerability. That said, the decision to grab it right away might depend on the level of immediate danger from the enemies; if it's too high, I might delay collecting the health until the nearest threats are neutralized.\n\nBased on the described scene, and depending on the precise dynamics of the enemies' locations and actions, I would typically ensure maintaining mobility, managing resources (health, ammo), and staying protected by cover would all be important factors, necessitating constant reassessment of the combat tactics being employed."
"Based on the context provided by the description, as a pro player, I would consider implementing the following strategic actions to effectively navigate the map and eliminate the enemies:\n\n1. **Attack action**: **Aim at the closest enemy that is not blocked**. The enemy directly in the line of sight (at around 12 o'clock but slightly to the left) is the most immediate threat and should be the focus of an attack. By aiming at this target, there would be a direct line of engagement. Following this, I'll proceed with the sub-action **Fire** to attempt to take out the enemy.\n\n2. **Reposition action**: **Sprint to 11 o'clock direction**. By moving towards the elevated position on the left, I can potentially gain a tactical advantage, potentially getting a clearer view of the area while reducing my exposure to enemy fire from the partially obscured enemy at 1 o'clock. This movement would get the player closer to the cover offered by the perpendicular wall mentioned in the description.\n\n3. **Other action**: **Collect health package within sight**. The package can ensure sustainability in the courtyard otherwise hazardous setting and provide the needed longevity to defeat the enemies in the immediate vicinity and move forward.\n\nThe exact sequence of actions after assessing the tactical situation might be:\n\n1. Collect health package within sight (ensuring I'm in optimal condition to engage and survive the battle).\n2. Aim at notch blocked enemy (the most immediate threat) and fire.\n3. Sprint to the 11 o'clock direction to potentially take cover and re-evaluate my position for the next tactics in taking out the remaining threats."

{Output2}
"Based on the described scenario and the choices provided, my actions as a pro player would be focused on preserving my health while efficiently dealing with the enemies. Here are the three actions I would take:\n\n1. Attack actions: Aim at the closest enemy that is not blocked. In this scenario, engaging the enemy directly in front would be optimal as they are not obscured and pose the most immediate threat.\n\n2. Attack actions: Fire. Once I have aimed at the closest enemy that is not blocked, I would immediately open fire to neutralize the threat. Quick reactions are key in a fast-paced game like DOOM.\n\n3. Reposition actions: Sprint in the direction of 2 o'clock. Since the description implies an exit or entryway to the player's right, sprinting towards 2 o'clock not only positions me to potentially use the exit/entry as a strategic advantage, but it can also help create a more favorable angle to deal with the second enemy who is slightly to the right. Engaging enemies from a new angle can provide a tactical advantage, and moving towards the exit/entry might open up escape routes or other strategic options.\n\nThese actions would, ideally, allow me to eliminate the immediate threat while also repositioning myself advantageously within the environment. Without more information on the presence of health packages, armor, or ammo within the immediate vicinity, it would be premature to take actions related to collecting those. Combat efficiency and positioning should take precedence in this context. After neutralizing the immediate threats, I would then reassess the environment to take further actions like collecting resources or finding a safe route to progress."

{Output3}
"Given the situation you've described from the DOOM gameplay perspective—considering we're a pro player focusing on navigation and enemy elimination in an environment that’s free of immediate threats—here are the three actions I would pick:\n\n1. **Reposition Action - Turn the camera view to direction of 1 ranging to 12 o'clock direction:** Firstly, I would turn the camera view towards the opening, specifically between the 11 to 1 o'clock positions. This step is crucial because it will give me a clearer overview of what's ahead, including potential enemy movement, opportunities for cover, or pickups that I might not have noticed yet. By centering my view on the doorway, I can prepare for an encounter or determine if the pathway is clear.\n\n2. **Other Action - Collect ammo within sight:** Depending on my current ammo status—which is critical in a fast-paced FPS like DOOM—I would collect any visible ammo around me. The description states that there's no immediate sighting of health packages or armor, so if there's ammo on the ground, particularly along my pathway towards that focal 12 o'clock position, ensuring that I'm fully stocked allows me to engage in a firefight without any mid-battle shortages.\n\n3. **Attack Action - Sprint in direction of 1 ranging to 12 o’clock direction:** Finally, if no immediate threat is noticeable and my primary objective is to move through the map quickly, I would select a sprint in the range of the 11 o'clock to 1 o'clock direction, specifically heading towards the 12 o'clock position through the central pathway. This rapid forward movement will close the distance to the next area and potential targets while also reducing my exposure to any threats that may arise behind me or from the sides.\n\nBy executing these three actions, I've ensured I have a good understand of my surroundings, am well-prepared in terms of ammunition, and I am moving swiftly to the next strategic point—all of which are quintessential to advancing through a DOOM level as a pro player."

{Output4}
'Considering the environment and enemies described, here are three strategic actions a pro player might take:\n\n1. Attack Action 2 (Aim at the closest enemy that is not blocked): The agent should prioritize eliminating the most immediate threat. Since one humanoid demon enemy is unobstructed and straight ahead (12 o\'clock), aiming at and engaging this enemy first would be the best course of action before they have a chance to advance or attack.\n\n2. Attack Action 4 (Fire): After aiming at the closest enemy, the next immediate action would be to fire upon the enemy. Ensuring accuracy and quickness will prevent the creature from closing the distance or retaliating, maximizing the player\'s chances of survival and maintaining control of the situation.\n\n3. Reposition Action 1 (Take cover behind the wall or obstacle in direction of 11 o\'clock): Since the player needs to stay mobile to avoid incoming attacks and possibly flanking maneuvers from the enemy in the background or others that might spawn, moving towards 11 o\'clock while keeping a line of sight to the corner of the structure/commanding wall offers strategic cover and the option to engage the second enemy. This action provides an optimal mix of defense and offense, allowing the player to protect themselves while planning their next move against other targets.\n\nThese actions use the \'DOOM\' gameplay philosophy of "push-forward combat" which encourages aggressive, fast-paced engagement rather than static gunfights from behind cover. The movement towards cover should be executed with precise timing and map knowledge to ensure continuous pressure on the enemies while staying relatively safe.'
