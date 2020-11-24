import gym
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
os.makedirs("outputs",exist_ok=True)

DEVICE="cuda:0"
ACTION_SPACE = [0,1]
EPISODES = 100000
STEPS = 1000
GAMMA=0.99
RENDER=False
def state_to_tensor( I):
        """ prepro 210x160x3 uint8 frame into 6000 (75x80) 1D float vector. See Karpathy's post: http://karpathy.github.io/2016/05/31/rl/ """
        if I is None:
            return torch.zeros(1, 6000)
        I = I[35:185] # crop - remove 35px from start & 25px from end of image in x, to reduce redundant parts of image (i.e. after ball passes paddle)
        I = I[::2,::2,0] # downsample by factor of 2.
        I[I == 144] = 0 # erase background (background type 1)
        I[I == 109] = 0 # erase background (background type 2)
        I[I != 0] = 1 # everything else (paddles, ball) just set to 1. this makes the image grayscale effectively
        return torch.from_numpy(I.astype(np.float32).ravel()).unsqueeze(0)
class ReinforceModel(nn.Module):
    def __init__(self):
        super(ReinforceModel,self).__init__()
        self.num_action = num_action
        self.num_input = num_input

        self.layers = nn.Sequential(
            nn.Linear(6000, 512), nn.ReLU(),
            nn.Linear(512, 2),
        )
        
    def forward(self,x):
        x = state_to_tensor(x)
        x = F.relu(self.layers(x))
        actions = F.softmax(self.layers(x))
        action = self.get_action(actions)
        log_prob_action = torch.log(actions.squeeze(0))[action]
        return action,log_prob_action
    def get_action(self,a):
        return np.random.choice(ACTION_SPACE,p=a.squeeze(0).detach().cpu().numpy())
    

env = gym.make('PongNoFrameskip-v4')
print(env.action_space,env.observation_space)

model = ReinforceModel().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
all_rewards =[]
best_rolling = -99999
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
        state,r_,done,i_ = env.step(action)
        lp.append(log_prob)
        r_ = r_ /100
        r.append(r_)
        if done:
            all_rewards.append(np.sum(r))
            
            if episode%100 ==0:
                print(f"EPISODE {episode} SCORE: {np.sum(r)} roll{pd.Series(all_rewards).tail(100).mean()}")
                # RENDER = True
                torch.save(model.state_dict(), 'outputs/last_params_cloud.ckpt')
                if pd.Series(all_rewards).tail(100).mean()>best_rolling:
                    best_rolling = pd.Series(all_rewards).tail(100).mean()
                    print("saving...")
                    torch.save(model.state_dict(), 'outputs/best_params_cloud.ckpt')
            break
 

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
