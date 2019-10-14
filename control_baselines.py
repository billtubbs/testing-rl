import logging
import numpy as np
import control
from gym_CartPole_BT.systems import cartpend


class LQR:
    """
    Linear quadratic Regulator (LQR) for use with Open AI Gym environments.

    LQR uses a simple linear feedback controller and is commonly used in
    control engineering applications.

    LQR works well when the system dynamics can be described by a set of
    ordinary differential equations around a fixed point and the cost is a
    quadratic function of the state.

    :param policy: (None) included for compatibility only.
    :param env: (Gym environment) the environment to control.
    :param gain: (array) gain matrix (K) of the linear controller.
    :param verbose: (int) the verbosity level: 0 to 2.
    :param requires_vec_env: For compatibility with Gym (Not used).
    :param policy_base: For compatibility with Gym (Not used).
    :param policy_kwargs: For compatibility with Gym (Not used).
    """

    def __init__(self, policy, env, gain=None, verbose=0, *,
                 requires_vec_env=False, policy_base=None, policy_kwargs=None):
        self.policy = policy
        self.env = env
        self.verbose = verbose
        self._requires_vec_env = requires_vec_env
        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs
        self.requires_vec_env=False
        if env is not None:
            self.observation_space = env.observation_space
            self.action_space = env.action_space
            gain_matrix_shape = (
                env.action_space.shape[0],
                env.observation_space.shape[0]
            )
            if gain is None:
                gain = np.zeros(gain_matrix_shape, dtype=float)
            else:
                gain = np.array(gain, dtype=float)
                assert gain.shape == gain_matrix_shape, "Gain matrix for " \
                    f"this environment should be shape {gain_matrix_shape}."
        self.gain = gain  # Gain matrix
        self.u = np.zeros(env.action_space.shape)  # Control vector

    def get_env(self):
        """returns the current environment (can be None if not defined)

        :return: (Gym Environment) The current environment
        """
        return self.env

    def set_env(self, env):
        """Checks the validity of the environment, and if it is coherent,
        set it as the current environment.
        :param env: (Gym Environment) The environment for learning a policy
        """

        # sanity checking the environment
        assert self.observation_space == env.observation_space, \
            "Error: the environment passed must have the same observation " \
            "space as this controller."
        assert self.action_space == env.action_space, \
            "Error: the environment passed must have the same action " \
            "space as this controller."

        self.env = env

    def predict(self, observation, state=None, mask=None, deterministic=True):

        observation = np.array(observation)
        self.u[:] = -np.dot(self.gain, observation - self.env.goal_state)

        return self.u, None


class LQRCartPend(LQR):

    def __init__(self, policy, env, gain=None, verbose=0,
                 _init_setup_model=True):

        super().__init__(policy, env, gain=gain, verbose=verbose,
                         requires_vec_env=False, policy_base=None)

        # State-space system parameters
        self.A = None
        self.B = None
        self.Q = None
        self.R = None
        if self.env is not None and _init_setup_model:
            self.setup_model()

    def setup_model(self):

        # Get cart-pendulum system parameters from environment
        m = self.env.masscart
        M = self.env.masspole
        L = self.env.length
        g = self.env.gravity
        d = self.env.friction
        goal_state = self.env.goal_state

        # Determine which fixed point to linearize at
        if np.array_equal(goal_state, np.array([0.0, 0.0, np.pi, 0.0])):
            s = 1
        elif np.array_equal(goal_state, np.array([0.0, 0.0, 0.0, 0.0])):
            s = -1
        else:
            raise ValueError("Linearizing at this goal state is not allowed.")

        # Calculate system and control matrices
        self.A, self.B = cartpend.cartpend_ss(m=m, M=M, L=L, g=g, d=d, s=s)

        # Choose state and input weights for cost function
        self.Q = np.eye(4)
        self.R = 0.0001

        # Derive the optimal controller
        K, S, E = control.lqr(self.A, self.B, self.Q, self.R)

        # Set gain matrix
        self.gain[:] = K


class BasicRandomSearch:
    """Basic random search (BRS) algorithm.

    A primitive form of random search which simply computes a finite 
    difference approximation along the random direction and then takes
    a step along this direction without using a line search.

    Refer to Mania et al. (2018) for details.
    """

    def __init__(self, policy_params, rollout_func, step_size=0.25, 
                 n_samples=10, noise_sd=1, rng=None):
        """
        Args:
            policy_params (np.ndarray): 1-d array containing initial
                values of the model parameters (can be zeros).
            rollout_func (function): Function that runs one roll-out in
                the environment.
            step_size (float): step-size (alpha).
            n_samples (int): Number of directions sampled per iteration (N).
            noise_sd (float): Standard deviation of the exploration noise (mu).
            rng (np.random.RandomState): Initialized random number generator.
        """

        self.theta = np.array(policy_params)
        assert len(self.theta.shape) == 1
        self.n_params = self.theta.shape[0]
        self.rollout = rollout_func
        self.step_size = step_size
        self.n_samples = n_samples
        self.noise_sd = noise_sd
        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng
        self.rollout_count = 0
    
    def search(self, n_iter=1):

        for _ in range(n_iter):
            
            # Sample n random directions from standard normal distribution
            delta_values = self.rng.randn(self.n_samples*self.n_params)\
                           .reshape((self.n_samples, self.n_params))
            
            # Run 2*n rollouts in each random direction
            cum_rewards = {'+': [], '-': []}
            for delta in delta_values:
                param_vector = self.theta + delta*self.noise_sd
                cum_reward = self.rollout(param_vector)
                self.rollout_count += 1
                cum_rewards['+'].append(cum_reward)
                param_vector = self.theta - delta*self.noise_sd
                cum_reward = self.rollout(param_vector)
                self.rollout_count += 1
                cum_rewards['-'].append(cum_reward)

            # Compute the finite difference approximation along the
            # random direction and update model parameters
            cum_rewards = {k: np.array(v) for k, v in cum_rewards.items()}
            dr = cum_rewards['+'] - cum_rewards['-']
            dr_dtheta = dr.reshape((self.n_samples, 1)) * delta_values     
            self.theta = self.theta + (self.step_size * dr_dtheta.sum(axis=0) / 
                                       self.n_samples)


class AugmentedRandomSearch:
    """Augmented random search (ARS) algorithm versions 1, 1-t, 2, and 2-t.

    An augmented version of basic random search where the update
    steps are scaling by the standard deviation of the rewards 
    collected at each iteration.

    Refer to Mania et al. (2018) for details.
    """

    def __init__(self, policy_params, rollout_func, step_size=0.25, 
                 n_samples=10, noise_sd=1, b=None, rng=None):
        """
        Args:
            policy_params (np.ndarray): 1-d array containing initial
                values of the model parameters (can be zeros).
            rollout_func (function): Function that runs one roll-out in
                the environment.
            step_size (float): step-size (alpha).
            n_samples (int): Number of directions sampled per iteration (N).
            noise_sd (float): Standard deviation of the exploration noise (mu).
            b (int): Number of top-performing directions to use (b < N).
                If b > N or b is None then b will be set to N.
            rng (np.random.RandomState): Initialized random number generator.
        """

        super().__init__(policy_params, rollout_func, step_size=step_size, 
                         n_samples=n_samples, noise_sd=noise_sd, rng=rng)
        self.b = b

    def search(self, n_iter=1):

        for _ in range(n_iter):
            
            # Sample n random directions from standard normal distribution
            delta_values = self.rng.randn(self.n_samples*self.n_params)\
                           .reshape((self.n_samples, self.n_params))
            
            # Run 2*n rollouts in each random direction
            cum_rewards = {'+': [], '-': []}
            for delta in delta_values:
                param_vector = self.theta + delta*self.noise_sd
                cum_reward = self.rollout(param_vector)
                self.rollout_count += 1
                cum_rewards['+'].append(cum_reward)
                param_vector = self.theta - delta*self.noise_sd
                cum_reward = self.rollout(param_vector)
                self.rollout_count += 1
                cum_rewards['-'].append(cum_reward)

            # Compute the finite difference approximation along the
            # random direction and update model parameters
            cum_rewards = {k: np.array(v) for k, v in cum_rewards.items()}
            dr = cum_rewards['+'] - cum_rewards['-']
            dr_dtheta = dr.reshape((self.n_samples, 1)) * delta_values     
            self.theta = self.theta + (self.step_size * dr_dtheta.sum(axis=0) / 
                                       self.n_samples)