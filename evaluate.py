import gym
import torch

from train import train
from util import obs_to_tensor, sample_act


def play(
    model,
    n_episodes,
    env_name='CartPole-v0',
    step_len=.02
):
    env = gym.make(env_name)
    
    for _ in range(n_episodes):
        x_t = env.reset()
        done = False
        t = 0
        
        while not done:
            env.render()
            with torch.no_grad():
                x_t = obs_to_tensor(x_t)
                _, log_pi_xt = model(x_t)
            
            a_t = sample_act(log_pi_xt)
            x_t, _, done, _ = env.step(a_t)
            t += 1
        print(f'ep_len: {t}')
    
    env.close()


if __name__ == '__main__':
    env_name = 'CartPole-v1'
    model = train(1000, 10, env_name)
    play(model, 30, env_name)
