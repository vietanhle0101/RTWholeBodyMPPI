import numpy as np
import concurrent.futures
import threading
from concurrent.futures import ThreadPoolExecutor
import mujoco
from scipy.interpolate import CubicSpline
from mujoco import rollout
import yaml

class BaseDPC:
    """
    Base class for Differentiable Predictive Control (DPC) controllers.
    """

    def __init__(self, model_path, config_path):
        # Load task-specific configurations
        with open(config_path, 'r') as file:
            params = yaml.safe_load(file)

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = params['dt']
        self.model.opt.enableflags = 1  # Override contact settings
        self.model.opt.o_solref = np.array(params['o_solref'])

        # MPC parameters
        self.horizon = params['horizon']
        self.h = params['dt']
        self.rollout_func = self.call_rollout
        self.cost_func = self.calculate_total_cost

        # Initialize rollouts
        self.state_rollouts = np.zeros(
            (self.horizon, mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_FULLPHYSICS.value))
        )

        # Action limits
        self.act_dim = 12
        self.act_max = np.array([0.863, 4.501, -0.888] * 4)
        self.act_min = np.array([-0.863, -0.686, -2.818] * 4)


    def call_rollout(self, initial_state, ctrl, state):
        """
        Perform a rollout of the model given the initial state and control actions.
        Args:
            initial_state (np.ndarray): Initial state of the model.
            ctrl (np.ndarray): Control actions to apply during the rollout.
            state (np.ndarray): State array to store the results of the rollout.
        """
        rollout.rollout(self.model, self.thread_local.data, skip_checks=True,
                        nroll=state.shape[0], nstep=state.shape[1],
                        initial_state=initial_state, control=ctrl, state=state)
        
