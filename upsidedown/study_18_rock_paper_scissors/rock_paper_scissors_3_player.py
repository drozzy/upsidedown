"""A simple multi-agent env with three agents playing rock paper scissors.

This demonstrates running the following policies in competition:
    (1) heuristic policy of repeating the same move
    (2) heuristic policy of beating the last opponent move
    (3) LSTM/feedforward PG policies
    (4) LSTM policy with custom entropy loss
"""

import random
from gym.spaces import Discrete

from ray import tune
# from ray.rllib.agents.pg.pg import PGTrainer
from ray.rllib.agents.pg.pg_torch_policy import PGTorchPolicy
from ray.rllib.policy.policy import Policy
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils import try_import_torch

torch = try_import_torch()

ROCK = 0
PAPER = 1
SCISSORS = 2


class RockPaperScissorsEnv(MultiAgentEnv):
    """Three-player environment for rock paper scissors.

    The observation is simply the last opponent action."""

    def __init__(self, _):
        self.action_space = Discrete(3)
        self.observation_space = Discrete(3)
        self.player1 = "player1"
        self.player2 = "player2"
        self.player3 = "player3"
        self.last_move = None
        self.num_moves = 0

    def reset(self):
        self.last_move = (0, 0, 0)
        self.num_moves = 0
        return {
            self.player1: self.last_move[0],
            self.player2: self.last_move[1],
            self.player3: self.last_move[2]
        }

    def step(self, action_dict):
        move1 = action_dict[self.player1]
        move2 = action_dict[self.player2]
        move3 = action_dict[self.player3]

        self.last_move = (move1, move2, move3)
        obs = {
            self.player1: self.last_move[0],
            self.player2: self.last_move[1],
            self.player3: self.last_move[2]
        }
        r1, r2, r3 = {
            (ROCK, ROCK, ROCK): (0, 0, 0),
            (ROCK, ROCK, PAPER): (-1, -1, 1),
            (ROCK, ROCK, SCISSORS): (0, 0, -1),
            
            (ROCK, PAPER, ROCK): (-1, 1, -1),
            (ROCK, PAPER, PAPER): (-1, 0, 0),
            (ROCK, PAPER, SCISSORS): (-1, -1, -1),
            
            (ROCK, SCISSORS, ROCK): (0, -1, 0),
            (ROCK, SCISSORS, PAPER): (-1, -1, -1),
            (ROCK, SCISSORS, SCISSORS): (1, -1, -1),

            (PAPER, ROCK, ROCK): (1, -1, -1),
            (PAPER, ROCK, PAPER): (0, -1, 0),
            (PAPER, ROCK, SCISSORS): (-1, -1, -1),

            (PAPER, PAPER, ROCK): (0, 0, -1),
            (PAPER, PAPER, PAPER): (0, 0, 0),
            (PAPER, PAPER, SCISSORS): (-1, -1, 1),

            (PAPER, SCISSORS, ROCK): (-1, -1, -1),
            (PAPER, SCISSORS, PAPER): (-1, 1, -1),
            (PAPER, SCISSORS, SCISSORS): (-1, 0, 0),

            (SCISSORS, ROCK, ROCK): (-1, 0, 0),
            (SCISSORS, ROCK, PAPER): (-1, -1, -1),
            (SCISSORS, ROCK, SCISSORS): (-1, 1, -1),

            (SCISSORS, PAPER, ROCK): (-1, -1, -1),
            (SCISSORS, PAPER, PAPER): (1, -1, -1),
            (SCISSORS, PAPER, SCISSORS): (0, -1, 0),

            (SCISSORS, SCISSORS, ROCK): (-1, -1, 1),
            (SCISSORS, SCISSORS, PAPER): (0, 0, -1),
            (SCISSORS, SCISSORS, SCISSORS): (0, 0, 0)
        }[(move1, move2, move3)]
        rew = {
            self.player1: r1,
            self.player2: r2,
            self.player3: r3
        }
        self.num_moves += 1
        done = {
            "__all__": self.num_moves >= 10,
        }
        return obs, rew, done, {}


class AlwaysSameHeuristic(Policy):
    """Pick a random move and stick with it for the entire episode."""

    def get_initial_state(self):
        return [random.choice([ROCK, PAPER, SCISSORS])]

    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        return list(state_batches[0]), state_batches, {}

    def learn_on_batch(self, samples):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass


# class BeatLastHeuristic(Policy):
#     """Play the move that would beat the last move of the opponents."""
#     def compute_actions(self,
#                         obs_batch,
#                         state_batches=None,
#                         prev_action_batch=None,
#                         prev_reward_batch=None,
#                         info_batch=None,
#                         episodes=None,
#                         **kwargs):
#         print(obs_batch)
#         raise ValueError
#         def successor(x):
#             if x[ROCK] == 1:
#                 return PAPER
#             elif x[PAPER] == 1:
#                 return SCISSORS
#             elif x[SCISSORS] == 1:
#                 return ROCK

#         return [successor(x) for x in obs_batch], [], {}

#     def learn_on_batch(self, samples):
#         pass

#     def get_weights(self):
#         pass

#     def set_weights(self, weights):
#         pass


def run_same_policy():
    """Use the same policy for both agents (trivial case)."""

    tune.run("PG", config={"env": RockPaperScissorsEnv})


def run_heuristic_vs_learned(use_lstm=False, trainer="PG"):
    """Run heuristic policies vs a learned agent.

    The learned agent should eventually reach a reward of ~5 with
    use_lstm=False, and ~7 with use_lstm=True. The reason the LSTM policy
    can perform better is since it can distinguish between the always_same vs
    beat_last heuristics.
    """

    def select_policy(agent_id):
        if agent_id == "player1":
            return "learned"
        elif agent_id == "player2":
            return "always_same"
        else:
            return "always_same"

    tune.run(
        trainer,
        stop={"timesteps_total": 400000},
        config={
            "env": RockPaperScissorsEnv,
            "gamma": 0.9,
            "num_workers": 4,
            "num_envs_per_worker": 4,
            "sample_batch_size": 10,
            "train_batch_size": 200,
            "multiagent": {
                "policies_to_train": ["learned"],
                "policies": {
                    "always_same": (AlwaysSameHeuristic, Discrete(3),
                                    Discrete(3), {}),
                    # "beat_last": (BeatLastHeuristic, Discrete(3), Discrete(3),
                    #               {}),
                    "learned": (None, Discrete(3), Discrete(3), {
                        "model": {
                            "use_lstm": use_lstm
                        }
                    }),
                },
                "policy_mapping_fn": select_policy,
            },
        })



if __name__ == "__main__":
    # run_same_policy()
    # run_heuristic_vs_learned(use_lstm=False)
    run_heuristic_vs_learned(use_lstm=True)
    # run_with_custom_entropy_loss()
