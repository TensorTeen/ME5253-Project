from collections import defaultdict
from functools import cached_property
from mpe2 import simple_spread_v3
from mpe2.simple_spread.simple_spread import Scenario
from mpe2._mpe_utils.core import Landmark, World, Entity, Agent
import numpy as np
import logging

from .environment import Environment

logger = logging.getLogger(__name__)

class CooperativeNavigationEnvironment(Environment):
    """
    Cooperative Navigation task (based on Lowe et al. (2017),
    using the Multi Particle Environments package for
    unspecified hyperparameters).
    """

    agent_size = 0.15
    landmark_size = 0.050
    n_states = None
    n_phi = None
    n_varphi = None
    # contradictory implementation to linear, due to continuous state space.
    # Leaving None to prevent confusion.
    # world size is [-1, 1] in both dimensions.

    def __init__(
        self,
        n_nodes=20,
        n_edges=None,
        max_cycles=10_000,
        seed=0,
    ):
        self.n_actions = 5
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.n_nodes = n_nodes
        # self.n_phi = n_phi
        # self.n_varphi = n_varphi
        self.n_edges = n_edges or int(2 * (n_nodes - 1))
        self.n_action_space = self.n_actions**n_nodes
        self.seed = seed
        self._env = simple_spread_v3.env(
            render_mode="rgb_array",
            N=n_nodes,
            max_cycles=max_cycles,
            dynamic_rescaling=False,
        )
        self._env.reset(seed=seed)
        self.log = defaultdict(list)
        self.reset()
        self._world: World = self._env.world
        self._scenario: Scenario = self._env.scenario
        self.landmarks: list[Landmark] = self._world.landmarks
        self.original_landmark_positions = np.array(
            [lm.state.p_pos for lm in self.landmarks]
        )
        self.agents: list[Agent] = self._world.agents
        self._agent_name_to_index = {ag.name: i for i, ag in enumerate(self.agents)}
        self.targets = rng.permutation(n_nodes).tolist()

    def reset(self):
        self._env.reset(seed=self.seed)
        self.n_step = 0
        for key in ("actions", "steps", "state", "reward"):
            if key in self.log:
                del self.log[key]

    @property
    def state(self):
        return self.agent_positions.flatten()

    def get_features(self, state, actions):
        return self.get_action_vec(state, actions), self.get_varphi(state)

    def get_action_vec(self, state, actions):
        # state_vec = self.agent_positions.flatten()
        actions_vec = np.zeros((self.n_nodes, self.n_actions))
        for i, a in enumerate(actions):
            actions_vec[i, a] = 1.0
        actions_vec = actions_vec
        # print(f"{state_vec = }, {actions_vec = }")
        return actions_vec

    def get_varphi(self, state):
        """
        Here, returns the state vector (agent positions)

        Parameters
        ----------
        state : int
            Just sent back directly

        Returns
        -------
        np.ndarray
            State vector (agent positions)
        """
        return state
    @cached_property
    def landmark_positions(self):
        return np.array([lm.state.p_pos for lm in self.landmarks])

    @property
    def agent_positions(self):
        return np.array([ag.state.p_pos for ag in self.agents])

    def reward(self, i):
        "Reward at a particular time-step."
        agent = self.agents[i]
        # Distance to target landmark
        target = self.landmark_positions[self.targets[i]]
        dist2 = np.sqrt(np.sum(np.square(agent.state.p_pos - target)))
        # agent.collide just says whether the agent is allowed to collide.
        collision = any(
            self._scenario.is_collision(agent, other)
            for other in self.agents
            if other is not agent
        )
        rew = -dist2 - int(collision)
        return rew

    def get_rewards(self, actions):
        """
        Get the rewards at the current time step. Does not take actions into account.

        Parameters
        ----------
        actions : Any
            Ignored.

        Returns
        -------
        np.ndarray
            Reward for each agent.
        """
        # self.next_step(actions)
        coeffs = self.rng.uniform(0, 2, self.n_nodes)
        # coeffs = np.ones(self.n_nodes)
        rewards = np.array([self.reward(i) for i in range(self.n_nodes)]) * coeffs
        logger.debug(f"{rewards = }")
        return rewards

    def next_step(self, actions):
        """Takes a step in the environment.

        Parameters:
        -----------
        * actions: list<int<n_actions>, n_nodes>
            List of actions for each agent.

        Returns:
        --------
        * observations: list<np.array, n_nodes>
            List of observations for each agent.
        * rewards: np.array<float, n_nodes>
            Array of rewards for each agent.
        * done: bool
            Whether the episode has ended.
        * info: dict
            Additional information.
        """
        assert (self.original_landmark_positions == self.landmark_positions).all(), (
            "Landmark positions have changed. This should not happen."
        )
        prev_pos = self.agent_positions.copy()
        actions = list(actions)
        for i in range(len(self.agents)):
            # landmark_prev = self.landmark_positions[
            #     self.targets[self._agent_name_to_index[agent.name]]
            # ]
            # action_space = self._env.action_space(agent.name)
            # mask = np.zeros(self.n_actions, dtype=np.int8)
            # mask[actions[self._agent_name_to_index[agent.name]]] = True
            # assert mask[actions[i]] == 1, f"{mask = }, {actions = }, {i = }"
            # action = action_space.sample(mask)
            self._env.step(actions[i])
            # The environment only changes after all agents have taken an action.
            # SimpleEnv only executes a world step when the next agent index is 0.
            obs, _, termination, truncation, info = self._env.last()
            # print(
            #     f"Landmark position moved from {landmark_prev} to {self.landmark_positions[self.targets[self._agent_name_to_index[agent.name]]]}"
            # )
            if termination or truncation:
                print(f"""
An agent has terminated. 
{type(self._env).mro() = }
{self._env.max_cycles = }
{self.get_rewards(actions) = }
{self.agent_positions = }
{self.landmark_positions = }
{self.targets = }
""")
        self.n_step += 1
        logger.debug(
            f""""
Previous positions: {prev_pos}
Actions taken:      {actions}
New positions:      {self.agent_positions}
"""
        )
        return None


if __name__ == "__main__":
    env = CooperativeNavigationEnvironment(n_nodes=10)
    # print(env.landmark_positions, env.agents, env.targets)
    print(env.next_step([4] * 10))
