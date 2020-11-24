import gym
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import ImageFont, ImageDraw, Image
import cv2
DEVICE="cuda:0"
ACTION_SPACE = [0,1,2,3]
STEPS = 1000
RENDER=True

class ReinforceModel(nn.Module):
    def __init__(self,num_action,num_input):
        super(ReinforceModel,self).__init__()
        self.num_action = num_action
        self.num_input = num_input

        self.layer1 = nn.Linear(num_input,64)
        self.layer2 = nn.Linear(64,num_action)
        
    def forward(self,x):
        x = torch.tensor(x,dtype=torch.float32,device=DEVICE).unsqueeze(0)
        x = F.relu(self.layer1(x))
        actions = F.softmax(self.layer2(x))
        action = self.get_action(actions)
        log_prob_action = torch.log(actions.squeeze(0))[action]
        return action,log_prob_action
    def get_action(self,a):
        return np.random.choice(ACTION_SPACE,p=a.squeeze(0).detach().cpu().numpy())
    
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

fig = plt.figure()
env = gym.make("LunarLander-v2")
print(env.action_space,env.observation_space)

model = ReinforceModel(4,8).to(DEVICE)
model.load_state_dict(torch.load("best_params_cloud.ckpt"))

model.eval()
ims = []
rewards = []
state = env.reset()
for step in range(STEPS):
    img = env.render(mode='rgb_array')
    action,log_prob = model(state)
        # print(action)
    state,reward,done,i_ = env.step(action)
    rewards.append(reward)
    # print(reward,done)
    cv2_im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im_rgb)

    draw = ImageDraw.Draw(pil_im)

    # Choose a font
    font = ImageFont.truetype("Roboto-Regular.ttf", 20)

    # Draw the text
    draw.text((0, 0), f"Step: {step} Action : {action} Reward: {int(reward)} Total Rewards: {int(np.sum(rewards))} done: {done}", font=font,fill="#FDFEFE")

    # Save the image
    img = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
    im = plt.imshow(img, animated=True)
    ims.append([im])
    if done:
        env.close()


                
        
        break

Writer = animation.writers['pillow']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
im_ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=3000,
                                    blit=True)
im_ani.save('ll_train1.gif', writer=writer)    