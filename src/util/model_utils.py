import io
import torch
from torch import nn
from util import CustomLogger


def build_network(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def save_models(model_filepath_dict: dict = {}) -> None:
    for model, filepath in model_filepath_dict.items():
        torch.save(model.state_dict(), f'./out/models/{filepath}')


def load_models(model_filepath_dict: dict = {}) -> None:

    for model, filepath in model_filepath_dict.items():
        with open(f'./out/models/{filepath}', 'rb') as f:
            buffer = io.BytesIO(f.read())
        state_dict = torch.load(buffer)
        model.load_state_dict(state_dict, strict=False)


def aggregate_models(state_dicts, empty_averaged_model) -> nn.Module:
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
