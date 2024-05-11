# Copyright (c) 2023 Michael Hu.
# This code is part of the book "The Art of Reinforcement Learning: Fundamentals, Mathematics, and Implementation with Python.".
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.

"""Evaluate the AlphaZero two agents on Dots."""
import sys
import torch
from network import AlphaZeroNet
from pipeline import create_mcts_player, set_seed, disable_auto_grad
from envs.dots import DotsGameEnv
from absl import flags

FLAGS = flags.FLAGS

# flags.DEFINE_string('black_ckpt', './checkpoints/dots/8x8/l3w8h8-f40-rb10-fcu80-gTrue-s11000.tar', 'Black Player')
flags.DEFINE_string('black_ckpt', './checkpoints/dots/8x8/i3.8.8-f40-rb10-fcu80-gTrue-s26000.ckpt', 'Black Player')
flags.DEFINE_string('red_ckpt', './checkpoints/dots/8x8/i3.8.8-f40-rb10-fcu80-gTrue-s64000.ckpt', 'Red Player')
flags.DEFINE_integer('seed', 1, 'Seed the runtime.')
flags.DEFINE_integer('games', 2, 'Games to play.')

# Initialize flags
FLAGS(sys.argv)


def main():
    set_seed(FLAGS.seed)

    runtime_device = 'cpu'
    if torch.cuda.is_available():
        runtime_device = 'cuda'

    def mcts_player_builder(ckpt_file, device):
        loaded_state = torch.load(ckpt_file, map_location=torch.device(device))
        network_config = loaded_state['network_config']
        input_shape = network_config['input_shape']
        _, width, height = input_shape

        mcts_config = loaded_state['mcts_config']
        c_puct_base = mcts_config['c_puct_base']
        c_puct_init = mcts_config['c_puct_init']
        num_simulations = mcts_config['num_simulations']

        num_parallel = mcts_config['num_parallel']

        network = AlphaZeroNet(
            input_shape,
            network_config['num_actions'],
            network_config['num_res_block'],
            network_config['num_filters'],
            network_config['num_fc_units'],
            network_config['gomoku']
        ).to(device)

        disable_auto_grad(network)

        network.load_state_dict(loaded_state['network'])

        network.eval()

        return create_mcts_player(
            network=network,
            device=device,
            num_simulations=num_simulations,
            num_parallel=num_parallel,
            root_noise=False,
            deterministic=True,
        ), width, height, c_puct_base, c_puct_init

    black_player, width, height, c_puct_base_black, c_puct_init_black = mcts_player_builder(FLAGS.black_ckpt, runtime_device)
    red_player, width, height, c_puct_base_red, c_puct_init_red = mcts_player_builder(FLAGS.red_ckpt, runtime_device)

    # Start to play game
    black_reward = 0
    red_reward = 0
    for game in range(FLAGS.games):
        eval_env = DotsGameEnv(width, height, False)
        done = False
        i = 0 if game % 2 == 0 else 1

        while not done:
            print ('.', end='')
            if i % 2 == 0:
                move, *_ = black_player(eval_env, None, c_puct_base_black, c_puct_init_black)
            else:
                move, *_ = red_player(eval_env, None, c_puct_base_red, c_puct_init_red)
            _, _, done, _ = eval_env.step(move)
            i += 1

        reward = eval_env.terminal_reward()
        if game % 2 == 0:
            if reward > 0:
                black_reward += reward
            else:
                red_reward += -reward
        else:
            if reward > 0:
                red_reward += reward
            else:
                black_reward += -reward

        print(f'\nBlack vs Red {black_reward}:{red_reward}. Games {game + 1}')


if __name__ == '__main__':
    main()
