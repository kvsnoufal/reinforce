import gym
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
DEVICE="cuda:0"
ACTION_SPACE = [0,1]
EPISODES = 800
STEPS = 500
GAMMA=0.9
RENDER=False
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
    

env = gym.make("CartPole-v0")

model = ReinforceModel(2,4).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
all_rewards =[]
for episode in range(EPISODES):
    done=False

    state = env.reset()
    lp=[]
    a=[]
    r=[]
    d=[]
    s=[]

    for step in range(STEPS):
        if RENDER:
            env.render()
        action,log_prob = model(state)
        # print(action)
        state,r_,done,i_ = env.step(action)
        lp.append(log_prob)
        r.append(r_)
        if done:
            all_rewards.append(np.sum(r))
            if episode%100 ==0:
                print(f"EPISODE {episode} SCORE: {np.sum(r)} roll{pd.Series(all_rewards).tail(30).mean()}")
            
            break
    discounted_rewards = np.zeros_like(r)

    discounted_rewards = []

    for t in range(len(r)):
        Gt = 0 
        pw = 0
        for r_ in r[t:]:
            Gt = Gt + GAMMA**pw * r_
            pw = pw + 1
        discounted_rewards.append(Gt)
    
    discounted_rewards = np.array(discounted_rewards)
        

          


    discounted_rewards = torch.tensor(discounted_rewards,dtype=torch.float32,device=DEVICE)
    discounted_rewards = (discounted_rewards - torch.mean(discounted_rewards))/ (torch.std(discounted_rewards))
    log_prob = torch.stack(lp)
    policy_gradient = -log_prob*discounted_rewards

    model.zero_grad()
    policy_gradient.sum().backward()
    optimizer.step()




import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import cv2
model.eval()
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

fig = plt.figure()
ACTION_SPACE = [0,1]
env = gym.make("CartPole-v0")
state = env.reset()
ims = []
rewards = []
for step in range(500):
    # env.render()
    img = env.render(mode='rgb_array')
    # print(img)
    action,log_prob = model(state)
    print(action)
    state,reward,done,_ = env.step(action)
    print(reward)
    rewards.append(reward)
    print(img.shape)
    cv2_im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im_rgb)

    draw = ImageDraw.Draw(pil_im)

    # Choose a font
    font = ImageFont.truetype("Roboto-Regular.ttf", 20)

    # Draw the text
    draw.text((0, 0), f"Step: {step} Action : {action} Reward: {reward} Total Rewards: {np.sum(rewards)} done: {done}", font=font,fill="#000000")

    # Save the image
    img = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
    # img = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2GRAY)
    im = plt.imshow(img, animated=True)
    ims.append([im])
env.close()    

Writer = animation.writers['pillow']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
im_ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=3000,
                                    blit=True)
im_ani.save('cp_train.gif', writer=writer)
    
