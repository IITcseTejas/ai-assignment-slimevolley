import gym
import slimevolleygym
from agents.minimax_agent import MinimaxAgent
import numpy as np

def main():
    env = gym.make("SlimeVolley-v0")
    agent = MinimaxAgent(env, depth=2)

    for episode in range(1):
        obs = env.reset()
        done = False

        while not done:
            env.render()

            action = agent.get_action(obs)  # Minimax chooses agent action
            opponent_action = env.action_space.sample()  # Random opponent

            joint_action = np.hstack([action, opponent_action])  # Combined action
            obs, reward, done, info = env.step(joint_action)

        print(f"Episode reward: {reward}")

    env.close()

if __name__ == "__main__":
    main()
