import io
from ppo.actor_critic import *
from gymnasium.spaces import Box, Discrete
import json
import base64
import torch
import numpy as np




class BaseModel:
    def __init__(self) -> None:
        super(BaseModel, self).__init__()


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
            json_dict = {}
            for key, value in state_dict.items():
                # Convert tensors to base64-encoded strings
                if isinstance(value, torch.Tensor):
                    value = value.cpu().numpy()
                    value = {
                        'dtype': str(value.dtype),
                        'shape': list(value.shape),
                        'data': base64.b64encode(value.tobytes()).decode('utf-8')
                    }
                    json_dict[key] = value
            return json.dumps(json_dict)

        p_state_dict = self.policy.p_net.state_dict()
        v_state_dict = self.value.v_net.state_dict()
        json_p_net = state_dict_to_json(p_state_dict)
        json_v_net = state_dict_to_json(v_state_dict)
        
        return json_p_net, json_v_net    
        
    def json_to_state_dicts(self, json_p_net, json_v_net):
        def json_to_state_dict(json_data):
            json_dict = json.loads(json_data)
            state_dict = {}
            for key, value in json_dict.items():
                dtype_str = value['dtype']
                dtype = getattr(np, dtype_str)
                shape = tuple(value['shape'])
                data = base64.b64decode(value['data'])
                data = np.frombuffer(data, dtype=dtype).reshape(shape)
                tensor = torch.tensor(data)
                state_dict[key] = tensor
            return state_dict
        p_net = json_to_state_dict(json_p_net)
        v_net = json_to_state_dict(json_v_net)
        return p_net, v_net