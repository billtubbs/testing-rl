import gym
import gym_CartPole_BT
from control_baselines import LQRCartPend

env_name = 'CartPole-BT-dH-vH-v0'
env = gym.make(env_name)

model = LQRCartPend(None, env, verbose=1)

# Display animated runs with model
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        s = input("Press enter to run again, 'q' to quit: ")
        if s.lower() == 'q':
            break
        obs = env.reset()

env.close()