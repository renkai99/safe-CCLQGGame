import pickle
import numpy as np
from scipy.linalg import block_diag
from Diff_robot_uncertainty import UnicycleRobotUncertain
from MultiAgentDynamics import MultiAgentDynamics
from trajectory_plotter import TrajectoryPlotter
from solvers.solver_pd import CLQGGpdSolver
from Montecarlo_test import MonteCarloTest
from Costs import ProximityCost

class Configuration:
    def __init__(self, scenario="intersection", dt=0.2, horizon=10, ref_cost_threshold=20, collision_dist=1.0, prob=0.95):
        self.dt = dt
        self.horizon = horizon
        self.ref_cost_threshold = ref_cost_threshold
        self.collision_dist = collision_dist
        self.prob = prob
        self.scenario = scenario
        self.identity_size = 4

        self._initialize_robots()
        self._initialize_dynamics()
        self._load_reference_trajectory()
        self._initialize_sigmas()
        self._initialize_prox_cost_list()

        self.Ps = np.zeros((self.mp_dynamics.num_agents, self.mp_dynamics.TIMESTEPS, 2, self.mp_dynamics.num_agents*4))
        self.alphas = np.zeros((self.mp_dynamics.num_agents, self.mp_dynamics.TIMESTEPS, 2))

        print(f"Configuration initialized for scenario: {self.scenario}")
        print(f"------------------------------------------------------------------------")


    def _initialize_robots(self):
        """Initialize robots based on the scenario."""
        sigma = [0.1, 0.1, 0.1, 0.1]
        if self.scenario == "intersection":
            x0_1 = [3, -1.0, np.pi, 1]
            x0_2 = [-3, 1.0, 0.0, 1]
            x0_3 = [-1, 4.0, -np.pi/2, 2.0]
            x_ref_1 = np.array([-2, -1.0, np.pi, 0])
            x_ref_2 = np.array([3, 1.0, 0, 0])
            x_ref_3 = np.array([-1, -3, -np.pi/2, 0])
        elif self.scenario == "lane_changing":
            x0_1 = [-3.0, -2.0, 0, 1.2]
            x0_2 = [-3.1, 2.0, 0, 1.1]
            x0_3 = [-3.0, 0.0, 0, 1.0]
            x_ref_1 = np.array([2, 2, 0, 0])
            x_ref_2 = np.array([2, -2, 0, 0])
            x_ref_3 = np.array([3, 0, 0, 0])
        else:
            raise ValueError(f"Scenario {self.scenario} not supported.")
        
        self.robot1 = UnicycleRobotUncertain(x0_1, x_ref_1, sigma, self.dt)
        self.robot2 = UnicycleRobotUncertain(x0_2, x_ref_2, sigma, self.dt)
        self.robot3 = UnicycleRobotUncertain(x0_3, x_ref_3, sigma, self.dt)

    def _initialize_dynamics(self):
        """Initialize the multi-agent dynamics."""
        self.mp_dynamics = MultiAgentDynamics(
            [self.robot1, self.robot2, self.robot3], self.dt, self.horizon, self.ref_cost_threshold, self.prob
        )
        self.costs = self.mp_dynamics.define_costs_lists(uncertainty=True)

    def _load_reference_trajectory(self):
        """Load the reference trajectory from a pickle file."""
        file_name = f'./Reference_trajectory/{self.scenario}_3car_ref.pkl'
        with open(file_name, 'rb') as f:
            parameters = pickle.load(f)
        self.xs = np.copy(parameters['xs'])
        self.control_inputs = np.copy(parameters['us'])
        self.prev_control_inputs = np.copy(parameters['us'])
        self.Acs, self.Bcs, self.As, self.Bs = self.mp_dynamics.get_linearized_dynamics_for_initial_state(
            self.xs, self.control_inputs
        )

    def _initialize_sigmas(self):
        """Initialize the uncertainty covariance matrices."""
        sigmas_block_diag = block_diag(*[np.zeros((self.identity_size, self.identity_size)) for _ in range(self.mp_dynamics.num_agents)])
        self.sigmas = np.array([sigmas_block_diag for _ in range(self.mp_dynamics.TIMESTEPS)])
        for i in range(self.mp_dynamics.num_agents):
            self.sigmas[0][i*self.identity_size:(i+1)*self.identity_size, i*self.identity_size:(i+1)*self.identity_size] = np.diag(
                [sigma for sigma in self.mp_dynamics.agent_list[i].uncertainty_params]
            )
        for ii in range(self.mp_dynamics.TIMESTEPS - 1):
            self.sigmas[ii + 1] = self.Acs[ii] @ self.sigmas[ii] @ self.Acs[ii].T

    def _initialize_prox_cost_list(self):
        """Initialize the proximity cost list for chance constraints."""
        self.prox_cost_list = [[] for _ in range(len(self.mp_dynamics.agent_list))]
        for i in range(len(self.mp_dynamics.agent_list)):
            for j in range(len(self.mp_dynamics.agent_list)):
                if i != j:
                    self.prox_cost_list[i].append(ProximityCost(self.collision_dist, i, j, 200.0))

    def get_algorithm(self, solver):
        """
        Initialize the constrained-LQG game solver.
        Other solvers coming soon (e.g., LCP, VI solvers).
        """
        if solver == 'PD':
            print("Using Primal-dual solver")
            return CLQGGpdSolver(self.mp_dynamics, self.prox_cost_list, self.sigmas, prob=self.prob)
        else:
            raise ValueError(f"Solver {solver} not supported.")

    def get_monte_carlo_tester(self, num_monte_carlo_tests):
        """Initialize the Monte Carlo safety tester."""
        return MonteCarloTest(
            self.mp_dynamics,
            num_monte_carlo_tests,
            self.collision_dist,
            self.costs,
            self.sigmas,
            np.zeros((self.mp_dynamics.num_agents, self.mp_dynamics.num_agents - 1, self.mp_dynamics.TIMESTEPS)),  # lambdas
            np.array([[[1] * self.mp_dynamics.TIMESTEPS] * (self.mp_dynamics.num_agents - 1)] * self.mp_dynamics.num_agents) * 0.005,  # Is
        )

    def get_plotter(self, xs, output_type='gif'):
        """Initialize the trajectory plotter."""
        return TrajectoryPlotter(xs, self.mp_dynamics, self.scenario, output_type)

