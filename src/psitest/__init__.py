"""PSITest: variational + finite-difference solvers for 1D Schrödinger problems."""

from .potentials import PotentialConfig, available_potentials, build_potential
from .models import ModelConfig, available_models, build_model, count_parameters
from .grid import GridConfig, make_grid
from .trainer import TrainConfig
from .solver_fd import fd_eigensolve_k
from .trace import RunContext
