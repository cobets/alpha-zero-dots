from dotsenv import DotsEnv, BLACK
from typing import Tuple
import numpy as np


class DotsGameEnv:
    def __init__(self) -> None:
        self.has_resign_move = False
        self.has_pass_move = False
        self.env = DotsEnv(8, 8)
        self.num_actions = self.env.action_length()
        self.action_dim = self.num_actions
        self.legal_actions = (self.env.board == 0).flatten().astype(int)
        self.to_play = 1
        self.opponent_player = -1
        self.steps = 0
        self.last_move = None
        self.last_player = None

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        if self.env.terminal():
            raise RuntimeError('Game is over, call reset before using step method.')
        if action is not None and not 0 <= int(action) <= self.env.action_length() - 1:
            raise ValueError(f'Invalid action. The action {action} is out of bound.')
        if action is not None and action not in self.env.legal_actions():
            raise ValueError(f'Illegal action {action}.')
        self.last_move = action
        self.last_player = 1 if self.env.player == BLACK else -1

        self.steps += 1
        reward = 0.0
        observation, step_reward, done = self.env.step(action)
        self.legal_actions = (self.env.board == 0).flatten().astype(int)
        self.to_play = 1 if self.env.player == BLACK else -1
        self.opponent_player = 1 if self.env.opponent == BLACK else -1
        if done:
            reward = float(self.env.terminal_reward())

        return observation, reward, done, {}

    def reset(self):
        self.env = DotsEnv(8, 8)
        self.legal_actions = (self.env.board == 0).flatten().astype(int)
        self.to_play = 1
        self.opponent_player = -1
        self.steps = 0
        self.last_move = None
        self.last_player = None

    def is_game_over(self):
        return self.env.terminal()

    def observation(self):
        return self.env.observation()

    def get_result_string(self) -> str:
        if not self.env.terminal():
            return ''
        reward = self.env.terminal_reward()
        if reward == 1:
            return 'B+1.0'
        elif reward == -1:
            return 'W+1.0'
        else:
            return 'DRAW'
