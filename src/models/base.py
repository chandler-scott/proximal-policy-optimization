import io
from ppo.actor_critic import *
from gymnasium.spaces import Box, Discrete
import json


class BaseModel:
    def __init__(self) -> None:
        super(BaseModel, self).__init__()
        print('base init')


    def create_networks(self, action_space, obs_dim, policy_hidden_sizes,
                        value_hidden_sizes, activation):
        # create policy
        # policy builder depends on action space..
        if isinstance(action_space, Box):
            self.policy = GaussianActor(
                obs_dim, action_space.shape[0], policy_hidden_sizes, activation)
        else:
            self.policy = CategoricalActor(
                obs_dim, action_space.n, policy_hidden_sizes, activation)
        # create value
        self.value = Critic(obs_dim, value_hidden_sizes, activation)

    def state_dicts_to_bytes(self):
        p_bytes_io = io.BytesIO()
        v_bytes_io = io.BytesIO()

        p_state_dict = self.policy.p_net.state_dict()
        v_state_dict = self.value.v_net.state_dict()

        p_bytes_io.seek(0)
        v_bytes_io.seek(0)

        # Convert state dictionaries to byte format and save them to BytesIO objects
        torch.save(p_state_dict, p_bytes_io)
        torch.save(v_state_dict, v_bytes_io)

        # Get the byte representations as strings
        p_bytes = p_bytes_io.getvalue()
        v_bytes = v_bytes_io.getvalue()

        return p_bytes, v_bytes

    def bytes_to_state_dicts(self, p_bytes, v_bytes):
        # Create BytesIO objects from the byte representations
        p_bytes_io = io.BytesIO(p_bytes)
        v_bytes_io = io.BytesIO(v_bytes)

        # Load the state dictionaries from the BytesIO objects
        p_state_dict = torch.load(p_bytes_io, map_location=torch.device('cpu'))
        v_state_dict = torch.load(v_bytes_io, map_location=torch.device('cpu'))

        return p_state_dict, v_state_dict

    def state_dicts_to_json(self):
        def state_dict_to_json(state_dict):
            layer_values = {}  # Dictionary to store values for each layer

            for name, param in state_dict.items():
                if '.' in name:
                    # Split the parameter name to get the layer name
                    layer_name = name.split('.')[0]
                else:
                    layer_name = name

                if layer_name not in layer_values:
                    layer_values[layer_name] = []

                # Convert the tensor values to a list and add them to the layer's value list
                layer_values[layer_name].append(param.tolist())

            # Convert layer values to JSON strings
            layer_values_json = {layer_name: json.dumps(values) for layer_name, values in layer_values.items()}

            return layer_values_json

        p_state_dict = self.policy.p_net.state_dict()
        v_state_dict = self.value.v_net.state_dict()
        json_p_net = state_dict_to_json(p_state_dict)
        json_v_net = state_dict_to_json(v_state_dict)
        
        return json_p_net, json_v_net    
        
    def json_to_state_dicts(self, json_p_net, json_v_net):
        def json_to_state_dict(layer_values_json):
            state_dict = {}

            for layer_name, values_json in layer_values_json.items():
                values = json.loads(values_json)
                tensors = [torch.tensor(value) for value in values]

                # If the layer has only one parameter, directly assign the tensor to the layer name
                if len(tensors) == 1:
                    state_dict[layer_name] = tensors[0]
                else:
                    # If the layer has multiple parameters, assign them with names like 'layer_name.weight', 'layer_name.bias', etc.
                    for i, tensor in enumerate(tensors):
                        param_name = f"{layer_name}.weight" if i == 0 else f"{layer_name}.bias"
                        state_dict[param_name] = tensor

            return state_dict
        p_net = json_to_state_dict(json_p_net)
        v_net = json_to_state_dict(json_v_net)
        return p_net, v_net