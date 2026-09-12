import os
from whole_body_mppi.utils.tasks import get_task
from whole_body_mppi.interface.simulator import Simulator
from whole_body_mppi.control.controllers.srbm_mppi import SRBM_MPPI

import argparse

def main(task, steps, headless, do_plot):
    T = steps

    SIMULATION_STEP = 0.01
    CTRL_UPDATE_RATE = 100

    # Soft contact model paramters
    TIMECONST = 0.02
    DAMPINGRATIO = 1.0
    
    # Get task data
    task_data = get_task(task)
    sim_path = os.path.join(os.path.dirname(__file__), "../whole_body_mppi", task_data["sim_path"])

    # Initialize agent and simulator
    agent = SRBM_MPPI(task=task)
    simulator = Simulator(agent=agent, viewer=not headless, T=T, dt=SIMULATION_STEP, timeconst=TIMECONST,
                          dampingratio=DAMPINGRATIO, model_path=sim_path, ctrl_rate=CTRL_UPDATE_RATE)
    
    # Run simulation
    simulator.run()
    if do_plot:
        simulator.plot_trajectory()

if __name__ == "__main__":
    # Define valid tasks
    VALID_TASKS = ['stairs', 'stand', 'walk_octagon', 'walk_straight', 'big_box',
                   'walk_octagon_hw', 'walk_straight_hw', 'stand_hw', 'climb_box_hw']

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run SRBM-MPPI in MuJoCo simulator.")
    parser.add_argument('--task', type=str, required=True, choices=VALID_TASKS, 
                        help=f"Name of the task. Must be one of {VALID_TASKS}.")
    parser.add_argument('--steps', type=int, default=1000, help="Simulation steps.")
    parser.add_argument('--headless', action='store_true', help="Run without MuJoCo viewer.")
    parser.add_argument('--plot', action='store_true', help="Plot trajectory after run.")
    args = parser.parse_args()

    # Run main with the provided task
    main(args.task, args.steps, args.headless, args.plot)
