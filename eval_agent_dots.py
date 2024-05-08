# Copyright (c) 2023 Michael Hu.
# This code is part of the book "The Art of Reinforcement Learning: Fundamentals, Mathematics, and Implementation with Python.".
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


"""Evaluate the AlphaZero agent on Dots."""
from absl import flags
import sys
import torch

from envs.dots import DotsGameEnv
from dots_pygame import main as pygame

FLAGS = flags.FLAGS
flags.DEFINE_string('white_ckpt', './checkpoints/dots/8x8/l3w8h8-f40-rb10-fcu80-gTrue-s11000.tar', 'Best model so far.')
# flags.DEFINE_string('white_ckpt', './checkpoints/dots/8x8/l3w8h8-f128-rb10-fcu128-gTrue-s39000.tar', 'Load the checkpoint file for white player.')
flags.DEFINE_integer('seed', 1, 'Seed the runtime.')

# Initialize flags
FLAGS(sys.argv)

from network import AlphaZeroNet
from pipeline import create_mcts_player, set_seed, disable_auto_grad


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
        # num_simulations = 500
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

    white_player, width, height, c_puct_base, c_puct_init = mcts_player_builder(FLAGS.white_ckpt, runtime_device)

    # Start to play game
    eval_env = DotsGameEnv(width, height, True)

    def ai_play():
        move, *_ = white_player(eval_env, None, c_puct_base, c_puct_init)
        _, _, done, _ = eval_env.step(move)

    pygame(eval_env, ai_play)


if __name__ == '__main__':
    main()
