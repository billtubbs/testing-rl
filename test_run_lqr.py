import gym
import gym_CartPole_BT
import numpy as np

# Create and initialize environment
env = gym.make('CartPole-BT-m2-dL-v0')
env.reset()

# Control vector (shape (1, ) in this case)
u = np.zeros(1)

# Open graphics window and draw animation
env.render()

# We will keep track of the cumulative rewards
cum_reward = 0.0

print(f"{'k':>3s}  {'u':>5s} {'reward':>6s} {'cum_reward':>10s}")
print("-"*28)

# Gain matrix (K) for optimal control
# (Calculated using lqr function with Q=np.eye(4), and R=0.0001)
gain = np.array([-100.00,   -197.54,   1491.28,    668.44])

# Run one episode
done = False
while not done:

    # Retrieve the system state
    x, x_dot, theta, theta_dot = env.state

    # Linear quadratic regulator
    # u[t] = -Ky[t]
    u[:] = -np.dot(gain, env.state - env.goal_state)

    # Run simulation one time-step
    observation, reward, done, info = env.step(u)

    # Update the animation
    env.render()

    # Process the reward
    cum_reward += reward

    # Print updates
    print(f"{env.time_step:3d}: {u[0]:5.1f} {reward:6.2f} {cum_reward:10.1f}")

input("Press enter to continue...")

# Close animation window
env.viewer.close()
