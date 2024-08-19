REGISTRY = {}

from .rnn_agent import RNNAgent
from .n_rnn_agent import NRNNAgent
from .rnn_ppo_agent import RNNPPOAgent
from .conv_agent import ConvAgent
from .ff_agent import FFAgent
from .central_rnn_agent import CentralRNNAgent
from .mlp_agent import MLPAgent
from .atten_rnn_agent import ATTRNNAgent
from .noisy_agents import NoisyRNNAgent
from .latent_rnn_agent import LatentCEDisRNNAgent
from .maic_agent import MAICAgent
from .rode_agent import RODEAgent
from .roco_agent import ROCOAgent
from .sr_agent import SRAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["n_rnn"] = NRNNAgent
REGISTRY["rnn_ppo"] = RNNPPOAgent
REGISTRY["conv_agent"] = ConvAgent
REGISTRY["ff"] = FFAgent
REGISTRY["central_rnn"] = CentralRNNAgent
REGISTRY["mlp"] = MLPAgent
REGISTRY["att_rnn"] = ATTRNNAgent
REGISTRY["noisy_rnn"] = NoisyRNNAgent
REGISTRY["latent_rnn_agent"] = LatentCEDisRNNAgent
REGISTRY['maic'] = MAICAgent
REGISTRY["rode"] = RODEAgent
REGISTRY["roco"] = ROCOAgent
REGISTRY["sr_agent"] = SRAgent