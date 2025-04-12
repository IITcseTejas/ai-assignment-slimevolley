import gym
import slimevolleygym
import numpy as np
from agents.minimax_agent import MinimaxAgent
from agents.alphabeta_agent import AlphaBetaAgent
import os

def record(agent_class, name, depth=2):
    video_dir = f"videos/{name}"
    os.makedirs(video_dir, exist_ok=True)

    env = gym.make("SlimeVolley-v0")
    env = gym.wrappers.RecordVideo(env, video_dir, episode_trigger=lambda x: True)

    agent = agent_class(env, depth=depth)

    obs = env.reset()
    done = False

    while not done:
        action = agent.get_action(obs)
        opponent_action = env.action_space.sample()
        joint_action = np.hstack([action, opponent_action])
        obs, reward, done, info = env.step(joint_action)
        env.render()

    print(f"[{name}] Episode reward:", reward)
    env.close()

if __name__ == "__main__":
    print("🎥 Recording Minimax Agent...")
    record(MinimaxAgent, "minimax", depth=2)

    print("🎥 Recording Alpha-Beta Agent...")
    record(AlphaBetaAgent, "alphabeta", depth=2)

    print("✅ Both videos saved in ./videos/")
