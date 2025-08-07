import numpy as np
from time import time
import matplotlib.pyplot as plt
from solve_lq_problem import solve_lq_game

class CLQGGpdSolver:
    def __init__(self, mp_dynamics, prox_cost_list, ref_traj_type, sigmas, prob, TOL_CC_ERROR=7e-2, phi=2.0):
        self.mp_dynamics = mp_dynamics
        self.prox_cost_list = prox_cost_list
        self.ref_traj_type = ref_traj_type
        self.sigmas = sigmas
        self.prob = prob
        self.TOL_CC_ERROR = TOL_CC_ERROR
        self.phi = phi

        # Initialize algorithm parameters
        self.iter = 0
        self.errors = [[[] for _ in range(mp_dynamics.num_agents - 1)] for _ in range(mp_dynamics.num_agents)]
        self.max_error = 100
        self.lambdas = np.zeros((mp_dynamics.num_agents, mp_dynamics.num_agents - 1, mp_dynamics.TIMESTEPS))
        self.mu = np.array([[[1] * mp_dynamics.TIMESTEPS] * (mp_dynamics.num_agents - 1)] * mp_dynamics.num_agents) * 0.01

        # (Optional) for augmented Lagrangian
        self.Is = np.array([[[1] * mp_dynamics.TIMESTEPS] * (mp_dynamics.num_agents - 1)] * mp_dynamics.num_agents) * 0.01        

    def run(self, xs, control_inputs, Ps, alphas, costs):
        """
        Executes the iterative constraint satisfaction algorithm.

        Parameters:
        - xs: State trajectories
        - control_inputs: Initial control inputs
        - Ps: Feedback parameters
        - alphas: Feedback scalars
        - costs: Cost functions
        """
        prev_control_inputs = control_inputs
        start_time = time()
        current_points = np.copy(xs)
        last_points = np.zeros_like(current_points)

        plt.ion()
        fig, ax = plt.subplots()
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.grid(True)
        colors = ['ro', 'go', 'bo', 'co', 'mo', 'yo']

        try:
            while self.max_error > self.TOL_CC_ERROR:
                self._update_errors(xs)
                self._update_lambdas()
                self._update_mu()
                if self.ref_traj_type == "known":
                    xs, control_inputs = self.mp_dynamics.compute_op_point(
                        Ps, alphas, xs, prev_control_inputs, 0.02, -1, False
                    )

                    Acs, Bcs, As, Bs = self.mp_dynamics.get_linearized_dynamics_for_initial_state(xs, control_inputs)
                    Gs, qs, rhos = self.mp_dynamics.get_Gs(xs, self.prox_cost_list, self.sigmas)
                    Qs, ls, Rs = self._compute_cost_matrices(xs, control_inputs, Gs, qs, rhos, costs)

                    Ps, alphas = solve_lq_game(As, Bs, Qs, ls, Rs)
                    prev_control_inputs = control_inputs

                    self.iter += 1
                else:
                    flag = False
                    while not flag:
                        self._update_errors(xs)
                        xs, control_inputs = self.mp_dynamics.compute_op_point(
                            Ps, alphas, xs, prev_control_inputs, 0.02, -1, False
                        )
                        ax.clear()
                        ax.grid(True)
                        ax.set_xlim(-4, 4)
                        ax.set_ylim(-4, 4)

                        # get the first elements of xs
                        for i in range(self.mp_dynamics.num_agents):
                            ax.plot([x[0] for x in xs[i]], [x[1] for x in xs[i]], colors[i], label=f'Robot {i}', markersize=5)

                        plt.pause(0.01)
                        plt.show()
                        last_points = np.copy(current_points)
                        current_points = np.copy(xs)

                        Acs, Bcs, As, Bs = self.mp_dynamics.get_linearized_dynamics_for_initial_state(xs, control_inputs)
                        Gs, qs, rhos = self.mp_dynamics.get_Gs(xs, self.prox_cost_list, self.sigmas)
                        Qs, ls, Rs = self._compute_cost_matrices(xs, control_inputs, Gs, qs, rhos, costs)

                        Ps, alphas = solve_lq_game(As, Bs, Qs, ls, Rs)
                        prev_control_inputs = control_inputs

                        self.iter += 1
                        if self.iter > 1:
                            flag = self.mp_dynamics.check_convergence(current_points, last_points)  

        except KeyboardInterrupt:
            print("Algorithm interrupted.")

        plt.ioff()
        plt.close()

        end_time = time()
        print(f"Number of iterations: {self.iter}")
        print(f"Solver time: {end_time - start_time:.2f} seconds")
        print(f"------------------------------------------------------------------------")
        return xs, prev_control_inputs, Ps, alphas,

    def _update_errors(self, xs):
        self.errors = [[[] for _ in range(self.mp_dynamics.num_agents - 1)] for _ in range(self.mp_dynamics.num_agents)]
        Gs, qs, rhos = self.mp_dynamics.get_Gs(xs, self.prox_cost_list, self.sigmas)

        for i, robot in enumerate(self.mp_dynamics.agent_list):
            for j in range(self.mp_dynamics.TIMESTEPS):
                for k in range(self.mp_dynamics.num_agents - 1):
                    concatenated_states = np.concatenate([state[j] for state in xs])
                    error = (Gs[i][j][k] @ concatenated_states + qs[i][j][k] + rhos[i][j][k])
                    self.errors[i][k].append(error)

        self.max_errors = np.float32(np.max(np.array(self.errors), axis=2))
        self.max_error = np.max(self.max_errors)

    def _update_lambdas(self):
        for i in range(self.mp_dynamics.num_agents):
            for j in range(self.mp_dynamics.num_agents - 1):
                for k in range(self.mp_dynamics.TIMESTEPS):
                    self.lambdas[i][j][k] = max(
                        0, self.lambdas[i][j][k] + self.mu[i][j][k] * (self.errors[i][j][k] - self.prob)
                    )
                    self.Is[i][j][k] = 0 if (self.prob - self.max_error < 0.0) and (self.lambdas[i][j][k] == 0) else self.mu[i][j][k]

    def _update_mu(self):
        for i in range(self.mp_dynamics.num_agents):
            for j in range(self.mp_dynamics.num_agents - 1):
                for k in range(self.mp_dynamics.TIMESTEPS):
                    self.mu[i][j][k] *= self.phi

    def _compute_cost_matrices(self, xs, control_inputs, Gs, qs, rhos, costs):
        Qs = [[] for _ in range(self.mp_dynamics.num_agents)]
        ls = [[] for _ in range(self.mp_dynamics.num_agents)]
        Rs = [[[] for _ in range(self.mp_dynamics.num_agents)] for _ in range(self.mp_dynamics.num_agents)]

        for t in range(self.mp_dynamics.TIMESTEPS):
            concatenated_states = np.concatenate([state[t] for state in xs])
            for i, robot in enumerate(self.mp_dynamics.agent_list):
                hessian_list = []
                gradient_list = []
                for j in range(self.mp_dynamics.num_agents - 1):
                    gradient_x = costs[i][0].gradient_x(
                        concatenated_states, control_inputs[i][t], Gs[i][t][j], qs[i][t][j], rhos[i][t][j],
                        self.lambdas[i][j][t], self.Is[i][j][t], timestep=t
                    )
                    hessian_x = costs[i][0].hessian_manual(
                        concatenated_states, control_inputs[i][t], Gs[i][t][j], qs[i][t][j], rhos[i][t][j],
                        self.lambdas[i][j][t], self.Is[i][j][t], timestep=t
                    )
                    hessian_list.append(hessian_x)
                    gradient_list.append(gradient_x)

                hessian_u = costs[i][0].hessian_u(concatenated_states, control_inputs[i][t])
                Qs[i].append(sum(hessian_list))
                ls[i].append(sum(gradient_list))
                Rs[i][i].append(hessian_u)

        for i in range(self.mp_dynamics.num_agents):
            for j in range(self.mp_dynamics.num_agents):
                if i != j:
                    Rs[i][j] = [np.zeros((2, 2)) for _ in range(self.mp_dynamics.TIMESTEPS)]

        return Qs, ls, Rs
