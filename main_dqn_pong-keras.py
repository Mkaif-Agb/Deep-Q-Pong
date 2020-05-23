import numpy as np 
from dqn import Agent
from utils import plotLearning, make_env
import os 
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


if __name__ == "__main__":
    env = make_env('PongNoFrameskip-v4')
    num_games = 10 
    load_checkpoint = True
    best_score = -21
    agent = Agent(gamma=0.99, epsilon=0.01, eps_decay=1e-5, alpha=0.0001, input_dims=(4, 80, 80),
                  n_actions=6, mem_size=35000, eps_min=0.01, batch_size=32, replace=1000)
    if load_checkpoint:
        agent.load_models()
    
    fname= 'keras-deep-q-pong/pong.png'
    scores = []
    eps_history = []
    n_steps = 0 
    for i in range(num_games):
        score = 0
        observation = env.reset()
        done = False
        while not done:
            env.render()
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            n_steps += 1 
            if not load_checkpoint:
                agent.store_transition(observation, action, reward, observation_, done)
                agent.learn()
            observation = observation_
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        print('Episode {} Score {} AverageScore {} Epsilon {} Steps {} '.format(i, score, avg_score, agent.epsilon, n_steps))
        if avg_score > best_score:
            agent.save_models()
            print('Average Score {} is better than best score {}'.format(avg_score, best_score))
            best_score = avg_score
        eps_history.append(agent.epsilon)
    
    x = [i+1 for i in range(num_games)]
    plotLearning(x, scores, eps_history, fname)
    
        