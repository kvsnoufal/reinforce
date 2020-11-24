import gym
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
os.makedirs("outputs",exist_ok=True)

DEVICE="cuda:0"
ACTION_SPACE = [0,1,2,3]
EPISODES = 100000
STEPS = 1000
GAMMA=0.99
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
    

env = gym.make("LunarLander-v2")
print(env.action_space,env.observation_space)

model = ReinforceModel(4,8).to(DEVICE)
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
