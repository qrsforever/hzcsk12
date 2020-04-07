import gym
import io
from gym import wrappers


# env = gym.make('SpaceInvaders-v0')
env = gym.make('Pendulum-v0')
print(env.action_space.sample())
# env = gym.make('CartPole-v0')
env = wrappers.Monitor(env, "/tmp/gym-results", force=True)
env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done: break
env.close()
