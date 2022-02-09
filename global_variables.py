import os

head = ["cycle",

        "rbc_center_position_x",
        "rbc_center_position_y",
        "rbc_center_position_z",

        "rbc_velocity_x",
        "rbc_velocity_y",
        "rbc_velocity_z",
        "rbc_cuboid_x_min_x",
        "rbc_cuboid_x_min_y",
        "rbc_cuboid_x_min_z",
        "rbc_cuboid_x_max_x",
        "rbc_cuboid_x_max_y",
        "rbc_cuboid_x_max_z",

        "rbc_cuboid_y_min_x",
        "rbc_cuboid_y_min_y",
        "rbc_cuboid_y_min_z",
        "rbc_cuboid_y_max_x",
        "rbc_cuboid_y_max_y",
        "rbc_cuboid_y_max_z",

        "rbc_cuboid_z_min_x",
        "rbc_cuboid_z_min_y",
        "rbc_cuboid_z_min_z",
        "rbc_cuboid_z_max_x",
        "rbc_cuboid_z_max_y",
        "rbc_cuboid_z_max_z",

        "rbc_cuboid_x_min_vel_x",
        "rbc_cuboid_x_min_vel_y",
        "rbc_cuboid_x_min_vel_z",
        "rbc_cuboid_x_max_vel_x",
        "rbc_cuboid_x_max_vel_y",
        "rbc_cuboid_x_max_vel_z",

        "rbc_cuboid_y_min_vel_x",
        "rbc_cuboid_y_min_vel_y",
        "rbc_cuboid_y_min_vel_z",
        "rbc_cuboid_y_max_vel_x",
        "rbc_cuboid_y_max_vel_y",
        "rbc_cuboid_y_max_vel_z",

        "rbc_cuboid_z_min_vel_x",
        "rbc_cuboid_z_min_vel_y",
        "rbc_cuboid_z_min_vel_z",
        "rbc_cuboid_z_max_vel_x",
        "rbc_cuboid_z_max_vel_y",
        "rbc_cuboid_z_max_vel_z",

        "volume",
        "surface",
        "NaN"
]

xy = [
        "rbc_center_position_x",
        "rbc_center_position_y",

        "rbc_velocity_x",
        "rbc_velocity_y",
        "rbc_cuboid_x_min_x",
        "rbc_cuboid_x_min_y",
        "rbc_cuboid_x_max_x",
        "rbc_cuboid_x_max_y",

        "rbc_cuboid_y_min_x",
        "rbc_cuboid_y_min_y",
        "rbc_cuboid_y_max_x",
        "rbc_cuboid_y_max_y",

        "rbc_cuboid_z_min_x",
        "rbc_cuboid_z_min_y",
        "rbc_cuboid_z_max_x",
        "rbc_cuboid_z_max_y",

        "rbc_cuboid_x_min_vel_x",
        "rbc_cuboid_x_min_vel_y",
        "rbc_cuboid_x_max_vel_x",
        "rbc_cuboid_x_max_vel_y",

        "rbc_cuboid_y_min_vel_x",
        "rbc_cuboid_y_min_vel_y",
        "rbc_cuboid_y_max_vel_x",
        "rbc_cuboid_y_max_vel_y",

        "rbc_cuboid_z_min_vel_x",
        "rbc_cuboid_z_min_vel_y",
        "rbc_cuboid_z_max_vel_x",
        "rbc_cuboid_z_max_vel_y",

        "volume",
        "surface",
]


xy_simple = [
        "rbc_center_position_x",
        "rbc_center_position_y",

        "rbc_velocity_x",
        "rbc_velocity_y",
        "rbc_cuboid_x_min_x",
        "rbc_cuboid_x_min_y",
        "rbc_cuboid_x_max_x",
        "rbc_cuboid_x_max_y",

        "rbc_cuboid_y_min_x",
        "rbc_cuboid_y_min_y",
        "rbc_cuboid_y_max_x",
        "rbc_cuboid_y_max_y",

        "rbc_cuboid_x_min_vel_x",
        "rbc_cuboid_x_min_vel_y",
        "rbc_cuboid_x_max_vel_x",
        "rbc_cuboid_x_max_vel_y",

        "rbc_cuboid_y_min_vel_x",
        "rbc_cuboid_y_min_vel_y",
        "rbc_cuboid_y_max_vel_x",
        "rbc_cuboid_y_max_vel_y",
]

#
xy_sized = [
        "rbc_cuboid_x_min_x",
        "rbc_cuboid_x_min_y",
        "rbc_cuboid_x_max_x",
        "rbc_cuboid_x_max_y",

        "rbc_cuboid_y_min_x",
        "rbc_cuboid_y_min_y",
        "rbc_cuboid_y_max_x",
        "rbc_cuboid_y_max_y",

        'rbc_velocity_x',
        'rbc_velocity_y',
        'rbc_cuboid_x_min_vel_x',
        'rbc_cuboid_x_min_vel_y',
        'rbc_cuboid_x_max_vel_x',
        'rbc_cuboid_x_max_vel_y',
        'rbc_cuboid_y_min_vel_x',
        'rbc_cuboid_y_min_vel_y',
        'rbc_cuboid_y_max_vel_x',
        'rbc_cuboid_y_max_vel_y',
        'x_x_size',
        'x_y_size',
        'y_x_size',
        'y_y_size',
]





xy_reduced = [
        "rbc_cuboid_x_min_x",
        "rbc_cuboid_x_min_y",
        "rbc_cuboid_x_max_x",
        "rbc_cuboid_x_max_y",

        "rbc_cuboid_y_min_x",
        "rbc_cuboid_y_min_y",
        "rbc_cuboid_y_max_x",
        "rbc_cuboid_y_max_y",

        'x_x_size',
        'y_y_size',
]

xy_reduced_standardize = [
        "rbc_cuboid_x_min_y",
        "rbc_cuboid_y_min_y",
        "rbc_cuboid_x_max_y",
        "rbc_cuboid_y_max_y",
        'x_x_size',
        'y_y_size',
]

xy_reduced_normalize = [
        "rbc_center_position_x",
        "rbc_cuboid_x_min_x",
        "rbc_cuboid_x_max_x",
        "rbc_cuboid_y_min_x",
        "rbc_cuboid_y_max_x",
]






xz_reduced = [
        "rbc_cuboid_x_min_x",
        "rbc_cuboid_x_min_z",
        "rbc_cuboid_x_max_x",
        "rbc_cuboid_x_max_z",

        "rbc_cuboid_z_min_x",
        "rbc_cuboid_z_min_z",
        "rbc_cuboid_z_max_x",
        "rbc_cuboid_z_max_z",

        'x_x_size',
        'z_z_size',
]

xz_reduced_standardize = [
        "rbc_cuboid_x_min_z",
        "rbc_cuboid_z_min_z",
        "rbc_cuboid_x_max_z",
        "rbc_cuboid_z_max_z",
        'x_x_size',
        'z_z_size',
]

xz_reduced_normalize = [
        "rbc_center_position_x",
        "rbc_cuboid_x_min_x",
        "rbc_cuboid_x_max_x",
        "rbc_cuboid_z_min_x",
        "rbc_cuboid_z_max_x",
]





xyz_reduced = [
        "rbc_cuboid_x_min_x",
        "rbc_cuboid_x_min_y",
        "rbc_cuboid_x_min_z",
        "rbc_cuboid_x_max_x",
        "rbc_cuboid_x_max_y",
        "rbc_cuboid_x_max_z",

        "rbc_cuboid_y_min_x",
        "rbc_cuboid_y_min_y",
        "rbc_cuboid_y_min_z",
        "rbc_cuboid_y_max_x",
        "rbc_cuboid_y_max_y",
        "rbc_cuboid_y_max_z",

        "rbc_cuboid_z_min_x",
        "rbc_cuboid_z_min_y",
        "rbc_cuboid_z_min_z",
        "rbc_cuboid_z_max_x",
        "rbc_cuboid_z_max_y",
        "rbc_cuboid_z_max_z",

        'x_x_size',
        'y_y_size',
        'z_z_size',
]

xyz_reduced_standardize = [
        "rbc_cuboid_x_min_y",
        "rbc_cuboid_x_min_z",
        "rbc_cuboid_x_max_y",
        "rbc_cuboid_x_max_z",

        "rbc_cuboid_y_min_y",
        "rbc_cuboid_y_min_z",
        "rbc_cuboid_y_max_y",
        "rbc_cuboid_y_max_z",

        "rbc_cuboid_z_min_y",
        "rbc_cuboid_z_min_z",
        "rbc_cuboid_z_max_y",
        "rbc_cuboid_z_max_z",

        'x_x_size',
        'y_y_size',
        'z_z_size',
]

xyz_reduced_normalize = [

        "rbc_cuboid_x_min_x",
        "rbc_cuboid_x_max_x",
        "rbc_cuboid_y_min_x",
        "rbc_cuboid_y_max_x",
        "rbc_cuboid_z_min_x",
        "rbc_cuboid_z_max_x",
]


SELECTED_AXIS = 'xy'
SELECTED_COLUMNS = xy_reduced
SELECTED_COLUMNS_TO_STANDARDIZE = xy_reduced_standardize
SELECTED_COLUMNS_TO_NORMALIZE = xy_reduced_normalize

# SELECTED_AXIS = 'xz'
# SELECTED_COLUMNS = xz_reduced
# SELECTED_COLUMNS_TO_STANDARDIZE = xz_reduced_standardize
# SELECTED_COLUMNS_TO_NORMALIZE = xz_reduced_normalize

# SELECTED_AXIS = 'xyz'
# SELECTED_COLUMNS = xyz_reduced
# SELECTED_COLUMNS_TO_STANDARDIZE = xyz_reduced_standardize
# SELECTED_COLUMNS_TO_NORMALIZE = xyz_reduced_normalize

# parameters

TS_LENGTH = 10
NUM_OF_RBC_TYPES = 3
LOSS_FN = 'mape'

START = 500
SAME_SIZE_OF_DF_FROM_SIMULATION = 2700
NUMBER_OF_AUGMENTATION = 30

SAVE_PATH = f'data/dataset/W_{TS_LENGTH}_A_{NUMBER_OF_AUGMENTATION}_X_{SELECTED_AXIS}'
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

STANDARDIZE = True

number_of_cells = 54

name_of_simulation = 'three_types'

EPOCHS = 1000
DROPOUT_RATE = .1
N_NODES = 512
LSTM_NODES = 256
