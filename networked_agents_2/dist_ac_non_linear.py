import logging

import numpy as np
import torch
from torch import nn

from .cooperative_environment import CooperativeNavigationEnvironment
from .dist_ac import DistributedActorCritic

logger = logging.getLogger(__name__)

class Predictor:
    def __init__(
        self,
        n_phi: int,
        n_nodes: int,
        n_hidden: int,
        n_actions: int,
        device: torch.device,
    ) -> None:
        self.device = device
        self.actor = nn.Sequential(
            nn.Linear(2 * n_nodes, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_actions),
            nn.Softmax(0),
        )
        self.critic = nn.Sequential(
            nn.Linear(n_phi, 1),
        )
        self.actor.to(self.device)
        self.critic.to(self.device)
        self._actor_optimizer = torch.optim.SGD(self.actor.parameters(), lr=0.01)
        self._critic_optimizer = torch.optim.SGD(self.critic.parameters(), lr=0.01)


class DistributedActorCriticNonLinear(DistributedActorCritic):
    n_hidden: int = 24
    env: CooperativeNavigationEnvironment

    def __init__(self, env: CooperativeNavigationEnvironment, device=None):
        self.env = env
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.n_actions = env.n_actions
        self.n_agents = env.n_nodes
        self.n_phi = 2 * env.n_nodes + env.n_actions
        self.mu = np.zeros(self.n_agents)
        # phi is the vector of [state, action]
        # super(DistributedActorCriticNonLinear, self).__init__(env)
        self.predictors: list[Predictor] = []
        for agent in range(self.n_agents):
            self.predictors.append(
                Predictor(
                    self.n_phi,
                    self.n_agents,
                    self.n_hidden,
                    self.n_actions,
                    self.device,
                )
            )
        self.n_steps = 0
        # Only for ensuring all gradients are zeroed

    def reset(self):
        self.n_steps = 0
        self.mu = np.zeros(self.n_agents)
        self.next_mu = np.zeros(self.n_agents)
        for predictor in self.predictors:
            predictor._actor_optimizer.zero_grad()
            predictor._critic_optimizer.zero_grad()
    def update(self, state, actions, rewards, next_state, next_actions, C):
        """Updates actor and critic parameters

        Parameters:
        -----------
        * state: tuple<np.array<n_phi>, np.array<n_actions, n_agents, n_varphi>>
            features representing the state where
            state[0]: phi represents the state at time t as seen by the critic.
            state[1]: varphi represents the state at time t as seen by the actor.
        * actions: np.array<n_agents>
            Actions for each agent at time t.
        * rewards: np.array<n_agents>
            Instantaneous rewards for each of the agents.
        * next_state: tuple<np.array<n_phi>, np.array<n_actions, n_agents, n_varphi>>
            features representing the state where
            next_state[0]: phi represents the state at time t+1 as seen by the critic.
            next_state[1]: varphi represents the state at time t+1 as seen by the actor.
        * next_actions: tuple(float<>, float<>)
            Actions for each agent at time t+1.
        """
        # NOTE - varphi should be replaced with phi for these functions, as the actions are handled on the output side, not the input layer
        # 1. Common knowledge at timestep-t
        action_vec, state_vec = self.env.get_features(state, actions)
        next_action_vec, next_state_vec = self.env.get_features(
            next_state, next_actions
        )
        # TODO: Won't actually be able to compute next_state, will need to figure that out
        rewards = rewards
        # dq = self.grad_q(phi)
        alpha = self.alpha
        beta = self.beta
        mu = self.mu
        advantages = []
        deltas = []
        grad_ws = []
        grad_thetas = []
        scores = []

        # wtilde = np.zeros_like(self.w)
        # 2. Iterate agents on the network.
        for i in range(self.n_agents):
            # 2.1 Compute time-difference delta
            with torch.no_grad():
                state_action_vec = np.concatenate((state_vec, action_vec[i]))
                next_state_action_vec = np.concatenate(
                    (next_state_vec, next_action_vec[i])
                )
                delta = (
                    rewards[i]
                    - mu[i]
                    + self.q(next_state_action_vec, i)
                    - self.q(state_action_vec, i)
                )
                # delta = np.clip(delta, -1.0, 1.0)
            with torch.no_grad():
                adv = self.advantage(
                    state_action_vec, state_vec, state, actions, i
                )  # [n_varphi,]
                # adv = np.clip(adv, -1.0, 1.0)
            predictor = self.predictors[i]
            predictor._critic_optimizer.zero_grad()
            predictor._actor_optimizer.zero_grad()
            q = self._q(state_action_vec, i)
            q.backward()

            # 2.2 Critic step
            # grad_w = alpha * delta * dq
            # wtilde[i, :] = self.w[i, :] + grad_w  # [n_phi,]

            # 3.3 Actor step
            # ksi = self.grad_log_policy(varphi, actions, i)  # [n_varphi,]
            prob = self._policy(state_vec, i)
            log_prob = torch.log(prob[actions[i]])
            log_prob.backward()
            torch.nn.utils.clip_grad_norm_(predictor.actor.parameters(), 0.10)
            torch.nn.utils.clip_grad_norm_(predictor.critic.parameters(), 0.10)
            for param in predictor.actor.parameters():
                param.data += param.grad * beta * adv
            # grad_theta = beta * adv * ksi
            # self.theta[i, :] += grad_theta  # [n_varphi,]

            # Log step
            # grad_thetas.append(grad_theta.tolist())
            # scores.append(ksi.tolist())

            advantages.append(adv.item())
            deltas.append(float(delta))

        # logger.debug(f"{C = }")
        # Consensus step.
        # w_tilde_params = []
        # for i in range(self.n_agents):
        #     critic = self.predictors[i].critic
        #     for param in critic.parameters():
        #         param.data += alpha * deltas[i] * param.grad
        #     w_tilde_params.append([param.data.clone() for param in critic.parameters()])
        # new_params = []
        C = np.eye((self.n_agents))
        for i in range(self.n_agents):
            # actor = self.predictors[i].actor
            critic = self.predictors[i].critic
            # for param in critic.parameters():
            #     param.data += alpha * deltas[i] * param.grad
            # for param in critic.parameters():
            #     # param.grad = None
            #     param.data = torch.zeros_like(param.data)
            for j in range(self.n_agents):
                if C[i, j] != 0:
                    other_critic = self.predictors[j].critic
                    for param, other_param in zip(
                        critic.parameters(), other_critic.parameters()
                    ):
                        param.data += C[i, j] * alpha * deltas[j] * other_param.grad
                        # print(other_param)
                        # param.data += C[i, j] * other_param
                # param.data = C[i, :] @ wtilde

        # Each agent is independent, so backward pass can be done independently
        # for param in predictor.actor.parameters():
        #     param.data += beta * delta * param.grad
        # Log.
        # ws.append(self.w.tolist())
        # thetas.append(self.theta.tolist())

        self.n_steps += 1
        self.mu = self.next_mu
        return advantages, deltas, scores

    def act(self, varphi):
        """Pick actions

        Parameters:
        -----------
        * varphi: [n_actions, n_agents, n_varphi]
            Critic features

        Returns:
        --------
        * actions: [n_agents,]
            Boolean array with agents actions for time t.
        """
        choices = []
        for i in range(self.n_agents):
            probs = self.policy(varphi, i)
            choices.append(int(np.random.choice(self.n_actions, p=probs)))
        return choices

    def update_mu(self, rewards):
        """Tracks long-term mean reward

        Parameters:
        -----------
        * rewards: float<n_agents>
            instantaneous rewards.
        """
        self.next_mu = (1 - self.alpha) * self.mu + self.alpha * rewards

    def advantage(self, phi, varphi, state, actions, i):
        return self.q(phi, i) - self.v(varphi, state, actions, i)
    def _q(self, phi: np.ndarray, i: int) -> torch.Tensor:
        """Q-function

        Parameters:
        -----------
        * phi: np.array<n_phi>
            critic features

        Returns:
        --------
        * q: float
            q-value for agent i
        """
        _phi = torch.tensor(phi, dtype=torch.float32).to(self.device)
        predictor = self.predictors[i]
        q = predictor.critic(_phi)
        return q

    def q(self, phi: np.ndarray, i: int):
        """Q-function

        Parameters:
        -----------
        * phi: np.array<n_phi>
            state, actions vector

        Returns:
        --------
        * q: float
           Q-value for agent i
        """
        with torch.no_grad():
            q = self._q(phi, i)
        return q.cpu().numpy()

    def get_q(self, phi: np.ndarray):
        return [self.q(phi, i).item() for i in range(self.n_agents)]

    def v(self, varphi, state, actions, i):
        """Relative value-function

        Value function computed as expected value of action-value function for policy specified by varphi

        Parameters:
        -----------
        * varphi: np.array<n_actions, n_agents, n_varphi>
            actor features
        * actions: np.array<n_agents>
            actions for agents
        * i: integer
            index of the agent on the interval {0,N-1}

        Returns:
        --------
        * v: float
            value-function with averaged i
        """
        probabilities = self.policy(varphi, i)
        actions_vec, state_vec = self.env.get_features(state, actions)
        action_values = np.array(
            [self.q(np.concatenate([state_vec, a]), i) for a in actions_vec]
        )
        return (probabilities * action_values).sum()

    def grad_q(self, phi):
        raise NotImplementedError

    def _policy(self, phi: np.ndarray, i: int) -> torch.Tensor:
        """
        Compute policy for state phi and agent i.
        """
        _phi = torch.tensor(phi, dtype=torch.float32).to(self.device)
        # print(_phi.shape)
        # print(f"{_phi = }")
        predictor = self.predictors[i]
        probs = predictor.actor(_phi)
        logger.debug(f"{probs = }")
        # print(f"Model parameters:\n{predictor.actor.state_dict()}")
        return probs

    def policy(self, varphi, i) -> np.ndarray:
        """Computes gibbs distribution / Boltzman policies

        Parameters:
        -----------
        * varphi: np.array<n_actions, n_agents, n_varphi>
            actor features


        * i: integer
            index of the agent on the interval {0,N-1}

        Returns:
        -------
        * probabilities: np.array<n_actions>
            Stochastic policy
        """
        with torch.no_grad():
            z = self._policy(varphi, i).cpu().numpy()

        return z

    def grad_log_policy(self, varphi, actions, i):
        raise NotImplementedError
