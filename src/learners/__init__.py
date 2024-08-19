from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .ppo_learner import PPOLearner
from .lica_learner import LICALearner
from .nq_learner import NQLearner
from .policy_gradient_v2 import PGLearner_v2
from .max_q_learner import MAXQLearner
from .dmaq_qatten_learner import DMAQ_qattenLearner
from .offpg_learner import OffPGLearner
from .fmac_learner import FMACLearner
from .latent_q_learner import LatentQLearner
from .maic_learner import MAICLearner
from .maic_qplex_learner import MAICQPLEXLearner
from .rode_learner import RODELearner
from .roco_learner import ROCOLearner
from .sr_learner import SRLearner
from .superQ import SuperQ
from .esip import Esip


REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["ppo_learner"] = PPOLearner
REGISTRY["lica_learner"] = LICALearner
REGISTRY["nq_learner"] = NQLearner
REGISTRY["policy_gradient_v2"] = PGLearner_v2
REGISTRY["max_q_learner"] = MAXQLearner
REGISTRY["dmaq_qatten_learner"] = DMAQ_qattenLearner
REGISTRY["offpg_learner"] = OffPGLearner
REGISTRY["fmac_learner"] = FMACLearner
REGISTRY["latent_q_learner"] = LatentQLearner
REGISTRY['maic_learner'] = MAICLearner
REGISTRY['maic_qplex_learner'] = MAICQPLEXLearner
REGISTRY["rode_learner"] = RODELearner
REGISTRY["roco_learner"] = ROCOLearner
REGISTRY["sr_learner"] = SRLearner
REGISTRY["superQ"] = SuperQ
REGISTRY["esip"] = Esip