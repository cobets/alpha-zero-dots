import torch

C_PUCT_BASE = 19652
C_PUCT_INIT = 1.25
NUM_SIMULATIONS = 380
NUM_PARALLEL = 8

INPUT_LEVELS = 3
WIDTH = 8
HEIGHT = 8
NUM_RES_BLOCK = 10
# NUM_FILTERS = 40
# NUM_FC_UNITS = 80
NUM_FILTERS = 128
NUM_FC_UNITS = 128

GOMOKU = True

if __name__ == '__main__':
    loaded_state = torch.load('./checkpoints/dots/8x8/training_steps_39000.ckpt', map_location=torch.device('cpu'))
    training_steps = loaded_state['training_steps']

    loaded_state['network_config'] = {
        'input_shape': (INPUT_LEVELS, WIDTH, HEIGHT),
        'num_actions': WIDTH * HEIGHT,
        'num_res_block': NUM_RES_BLOCK,
        'num_filters': NUM_FILTERS,
        'num_fc_units': NUM_FC_UNITS,
        'gomoku': GOMOKU
    }

    loaded_state['mcts_config'] = {
        'c_puct_base': C_PUCT_BASE,
        'c_puct_init': C_PUCT_INIT,
        'num_simulations': NUM_SIMULATIONS,
        'num_parallel': NUM_PARALLEL
    }

    network_file_name = f'l{INPUT_LEVELS}w{WIDTH}h{HEIGHT}-f{NUM_FILTERS}-rb{NUM_RES_BLOCK}-fcu{NUM_FC_UNITS}-g{GOMOKU}-s{training_steps}.tar'

    torch.save(loaded_state, f'./checkpoints/dots/8x8/{network_file_name}')
