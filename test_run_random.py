import gym
import gym_CartPole_BT
import numpy as np
import pandas as pd

def run_episode(env, agent, render=True, show=True):

    env.reset()

    if render:
        # Open graphics window and draw animation
        env.render()

    if show:
        print(f"{'k':>3s}  {'u':>5s} {'reward':>6s} "
               "{'cum_reward':>10s}")
        print("-"*28)

    # Keep track of the cumulative rewards
    cum_reward = 0.0

    # Run one episode
    done = False
    while not done:

        # Retrieve the system state
        x, x_dot, theta, theta_dot = env.state

        # Linear quadratic regulator
        u = agent.action(env)

        # Run simulation one time-step
        observation, reward, done, info = env.step(u)

        if render:
            # Update the animation
            env.render()

        # Process the reward
        cum_reward += reward

        if show:
            # Print updates
            print(f"{env.time_step:3d}: {u[0]:5.1f} "
                  f"{reward:6.2f} {cum_reward:10.1f}")

    return cum_reward


class ControllerLQR:
    """Linear quadratic regulator"""

    def __init__(self, gain):

        self.gain = gain
        self.u = np.zeros(1)

    def action(self, env):

        # Control vector (shape (1, ) in this case)
        self.u[0] = -np.dot(self.gain, env.state - env.goal_state)

        return self.u


# Create and initialize environment
env_name = 'CartPole-BT-dL-vL-v0'
print(f"\nInitializing environment '{env_name}'...")
env = gym.make(env_name)

# Use random search to find the best linear controller:
# u[t] = -Ky[t]

search_size = 1000.0

search_parameters = {
    'CartPole-BT-v0': (1, 1),
    'CartPole-BT-dL-v0': (100, 3),
    'CartPole-BT-vL-v0': (100, 3),
    'CartPole-BT-dL-vL-v0': (100, 3),
    'CartPole-BT-dH-v0': (500, 5),
    'CartPole-BT-vH-v0': (1000, 5)
}

try:
    n_iter, top_n = search_parameters[env_name]
except:
    n_iter, top_n = (100, 3)

# Start random search over full search area
print(f"\nStarting random search for {n_iter} episodes...")
results = []
for i in range(n_iter):
    gain = (np.random.random(size=4) - 0.5)*search_size
    agent = ControllerLQR(gain)
    cum_reward = run_episode(env, agent, render=False, show=False)
    results.append((cum_reward, gain))

top_results = pd.DataFrame(results, columns=['cum_reward', 'gain'])
top_results = top_results.sort_values(by='cum_reward', ascending=False)
top_results = top_results.reset_index(drop=True).head(top_n)
print(f"Top {top_n} results after {n_iter} episodes:\n"
      f"{top_results[['cum_reward']].round(2)}")

top_gains = np.vstack(top_results['gain'].values)
mean_gain = np.mean(top_gains, axis=0)
std_gain = top_gains.std(axis=0)

df = pd.DataFrame({'Mean': mean_gain, 'Std.': std_gain})
print(f"Mean gain values and std. dev:\n{df.round(2)}")

print(f"\nStarting targetted search for {n_iter} episodes...")
# Now search within reduced area
for i in range(n_iter):
    gain = np.random.normal(mean_gain, std_gain)
    agent = ControllerLQR(gain)
    # Average of two runs
    cum_reward = run_episode(env, agent, render=False, show=False)
    #print(f"{i}: Cum reward: {cum_reward}")
    results.append((cum_reward, gain))

top_results = pd.DataFrame(results, columns=['cum_reward', 'gain'])
top_results = top_results.sort_values(by='cum_reward', ascending=False)
top_results = top_results.reset_index(drop=True).head(top_n)
print(f"Top {top_n} results after {n_iter} episodes:\n"
      f"{top_results[['cum_reward']].round(2)}")

print(f"\nStarting robustness checks on top {top_n} results...")
results = []
# Do a robustness check on top results:
for gain in top_results['gain']:
    agent = ControllerLQR(gain)
    # Average over 3 runs
    mean_reward = np.mean([
        run_episode(env, agent, render=False, show=False)
        for _ in range(3)
    ])
    results.append(mean_reward)

best = np.argmax(results)
best_reward, best_gain = top_results.loc[best]

print(f"\nBest result (#{best}):")
print(f" Reward: {round(best_reward, 2)}")
print(f" Gain: {best_gain.round(3)}")

input("\nPress enter to start simulation...")

agent.gain = best_gain
cum_reward = run_episode(env, agent, render=True, show=False)
print(f"Reward: {round(cum_reward, 2)}")

# Close animation window
env.viewer.close()
