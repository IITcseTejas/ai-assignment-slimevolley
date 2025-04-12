import gym
import slimevolleygym

env = gym.make("SlimeVolley-v0")
obs = env.reset()

done = False
while not done:
    obs, reward, done, info = env.step(env.action_space.sample())
    env.render()

env.close()
