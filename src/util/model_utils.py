import io
import gymnasium as gym
import numpy as np
import pickle
import torch
from torch import nn
from util import CustomLogger

def to_doubles_list(py_ndarray):
    np_array = np.array(py_ndarray)
    float_list = np_array.tolist()
    return float_list

def state_dict_to_bytes(state_dict):
    bytes_io = io.BytesIO()

    # Convert state dictionaries to byte format and save them to BytesIO objects
    torch.save(state_dict, bytes_io)

    # Get the byte representations as strings
    bytes = bytes_io.getvalue()

    return bytes

def bytes_to_state_dict(bytes):
    # Create BytesIO objects from the byte representations
    bytes_io = io.BytesIO(bytes)

    # Load the state dictionaries from the BytesIO objects
    state_dict = torch.load(bytes_io, map_location=torch.device('cpu'))

    return state_dict

def serialize_state_dict(state_dict):
    serialized_state_dict = pickle.dumps(state_dict)
    return serialized_state_dict

def deserialize_state_dict(serialized_state_dict):
    state_dict = pickle.loads(serialized_state_dict)
    return state_dict

def to_tensor(init_values, lower_bound=-1, upper_bound=1):
    try:
        # Check if the provided bounds are valid
        if lower_bound >= upper_bound:
            raise ValueError("Lower bound must be less than the upper bound.")

        # Convert the list of floats to a NumPy array
        init_values_array = np.array(init_values, dtype=np.float32)

        # Normalize the values between 0 and 1 based on the bounds
        normalized_values = (init_values_array - lower_bound) / (upper_bound - lower_bound)

        # Convert the NumPy array to a PyTorch tensor
        tensor_values = torch.tensor(normalized_values, dtype=torch.float32)

        return tensor_values

    except Exception as e:
        print("An error occurred:", e)
        return None  # You can decide how to handle the error; returning None as an example
    
def zeros_box_space(box_size, lower_bound=-1, upper_bound=1):
    # Check if the provided bounds are valid
    if lower_bound >= upper_bound:
        raise ValueError("Lower bound must be less than the upper bound.")

    # Create the Box space initialized with zeros
    box_space = gym.spaces.Box(low=lower_bound, high=upper_bound, shape=(box_size,), dtype=np.float32)

    return box_space

def build_network(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def save_models(model_filepath_list: list = []) -> None:
    for model, filepath in model_filepath_list:
        torch.save(model, f'./models/{filepath}')


def load_models(model_filepath_dict: dict = {}) -> None:

    for model, filepath in model_filepath_dict.items():
        with open(f'./models/{filepath}', 'rb') as f:
            buffer = io.BytesIO(f.read())
        state_dict = torch.load(buffer)
        model.load_state_dict(state_dict, strict=False)


def aggregate_models(state_dicts:list, empty_averaged_model) -> nn.Module:
    ''' 
    Aggregate neural networks by taking the 
    mean of each parameter and weight 
    '''

    # Get the parameters of the first model
    average_params = list(state_dicts[0].values())

    # Iterate over the remaining models and add their parameters to average_params
    for model in state_dicts[1:]:
        for i, param in enumerate(model.values()):
            average_params[i] += param

    # Divide each parameter by the number of models to get the average
    for i in range(len(average_params)):
        average_params[i] /= len(state_dicts)

    avg_model = empty_averaged_model

    # Copy the state of the first model
    avg_model.load_state_dict(state_dicts[0])

    # Update the averaged model's state_dict with the average parameters
    avg_model_state_dict = avg_model.state_dict()
    for name, param in zip(avg_model_state_dict, average_params):
        avg_model_state_dict[name].copy_(param)

    return avg_model


def print_model(network, message='Neural Network Printout'):
    '''
    Print the neural network for debugging purposes
    '''
    logger = CustomLogger()
    logger.debug(f"{message}")
    for i, layer in enumerate(network):
        if isinstance(layer, torch.nn.Linear):
            # Print the weight and bias of each Linear layer
            logger.debug(f"Layer {i}:")
            logger.debug(f"Weight: {layer.weight}")
            logger.debug(f"Bias: {layer.bias}")
        elif isinstance(layer, torch.nn.Tanh):
            logger.debug(f"Layer {i}: Tanh activation")
        elif isinstance(layer, torch.nn.Identity):
            logger.debug(f"Layer {i}: Identity activation")
        else:
            logger.debug(f"Layer {i}: Unknown layer type")
