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
model.load_state_dict(torch.load("pong_best_params_cloud_1day.ckpt"))
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
im_ani.save('pong_train1.gif', writer=writer)    