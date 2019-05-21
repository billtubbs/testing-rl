# testing-rl
Scripts to test various reinforcement learning algorithms on simple control problems.

***This is a work-in-progress...***

To start off, I developed a new class that implements a [linear quadratic regulator](https://en.wikipedia.org/wiki/Linear–quadratic_regulator) (LQR).  This is a class of controllers used in control engineering applications to stabilize [dynamical systems](https://en.wikipedia.org/wiki/Dynamical_system). 

To find the optimal controller for a given dynamical system you ideally need to know the system dynamics.  The following module uses the [Python Control Systems Library](https://python-control.readthedocs.io/en/0.8.2/) to find an LQR for a cart-pendulum system:

- [control_baselines.py](control_baselines.py)

Now, I'm trying out some of the [Stable Baselines](https://stable-baselines.readthedocs.io/en/master/) implementations of reinforcement learning (RL) algorithms.

I'm testing these on some new environments I created in OpenAI's [Gym](https://gym.openai.com) toolkit.  They are a bit more challenging than the 'classic control' problems on OpenAI's website but east enough to understand.

To check these new environments out, see my other repo:
- [gym-CartPole-bt-v0](https://github.com/billtubbs/gym-CartPole-bt-v0)

For a quick demo, run one of the following script:
- [`test_run.py`](test_run.py) - this is a cart-pole environment simulated with no control inputs
- [`test_run_lqr.py`](test_run_lqr.py) - this is a cart-pole simulated with an optimal controller


