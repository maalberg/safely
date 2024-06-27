import numpy as np
import cvxpy as cp

import tensorflow as tf
import gpflow

import function as fun

class policy_iter:
    def __init__(
            self,
            policy: fun.function, dynamics: fun.function,
            reward: fun.function, value: fun.function,
            discount: tf.Tensor = tf.constant(0.98, dtype=gpflow.default_float())) -> None:

        self._policy = policy
        self._dynamics = dynamics

        self._reward = reward
        self._value = value
        self._discount = discount

    def _evaluate_opt(self, states_next: np.ndarray, rewards: np.ndarray) -> np.ndarray:
        """
        Evaluate current policy with ``states_next`` and ``rewards`` and
        return optimized value function parameters.

        The Bellman's equation V(s) = R + gamma*P*V(s') can be viewed as a
        linear program y = mx + c, where rewards R are constants c,
        value function V(s) are variables x, a probability
        transition matrix P is slope m.

        So to find optimal values for x we can solve this linear program using
        convex optimization techniques.
        """

        # values are optimization variables
        values = cp.Variable(rewards.shape)

        # Create a transition matrix, which characterizes the probability of transitioning
        # from current states to their next states, or s' = A * s,
        # where A is the transition matrix, and where s' and s
        # are the next and current states, respectively.
        values_derivative = self._value.parameters_derivative(states_next)
        values_derivative = cp.Constant(values_derivative)

        # maximizing the sum of future values shows the utility of choosing particular values
        utility = cp.Maximize(cp.sum(values))

        # constraint is composed of bellman's equation
        constraints = [
            values <= rewards + self._discount * values_derivative @ values]

        # construct optimization problem and solve it
        opt = cp.Problem(utility, constraints)
        opt.solve()

        # return optimal values
        return np.array(values.value)

    def evaluate(self, states: tf.Tensor, actions: tf.Tensor = None, optimize: bool = False) -> tf.Tensor:
        """
        Evaluate policy on given ``states`` and ``actions`` to return an updated array of values.
        Provided flag ``optimize`` is set to True, convex optimization is used
        to find an optimal value function.
        """

        if actions is None:
            # there are no given actions, so apply current policy to
            # produce actions according to given states
            actions = self._policy(states)

        # compose domain from states and actions in order to evaluate dynamics
        domain = tf.concat([states, actions], axis=1)

        # knowing dynamics, use states and actions to move to the next states
        states_next = self._dynamics(domain)

        # compute rewards for taking these actions given current states
        rewards = self._reward(domain)

        if optimize:
            return self._evaluate_opt(states_next, rewards)

        # predict future values and update them based on rewards and discount factor
        return rewards + self._discount * self._value(states_next)

    def improve(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """
        Produce an improvement update for the underlying policy by evaluating ``states``
        with every action from ``actions`` space
        """

        # query data dimensions
        states_n = states.shape[0]
        actions_n, actions_dims_n = actions.shape

        # prepare an array to hold future values evaluated for every action
        # in the given action-space
        values_eval = np.empty((states_n, actions_n))

        # prepare an array version of action to evaluate
        actions_eval = np.zeros((states_n, actions_dims_n))

        for this, action in enumerate(actions):
            # evaluated action [the whole array in fact] is set to
            # the current action from the given action-space
            actions_eval[:] = action

            # store values evaluated for every action
            values_eval[:, this] = self.evaluate(states, actions_eval).squeeze()

        # optimal actions are selected for every state,
        # so if we have, say, three possible actions to perform,
        # then one action is selected, which maximizes the long-term expected value for a particular state
        return actions[np.argmax(values_eval, axis=1)]
