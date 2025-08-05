# monte_carlo_safety_test.py
import numpy as np
from Costs import ProximityCost

class MonteCarloTest:
    def __init__(self, mp_dynamics, num_tests, collision_dist, costs, sigmas, lambdas, Is):
        self.mp_dynamics = mp_dynamics
        self.num_tests = num_tests
        self.collision_dist = collision_dist
        self.costs = costs
        self.Is = Is
        self.sigmas = sigmas
        self.lambdas = lambdas
        self.num_coll = 0
        self.total_costs = []

        self.prox_cost_list = [[] for _ in range(len(mp_dynamics.agent_list))]
        for i in range(len(mp_dynamics.agent_list)):
            for j in range(len(mp_dynamics.agent_list)):
                if i != j:
                    self.prox_cost_list[i].append(ProximityCost(collision_dist, i, j, 200.0))

    def run_tests(self, Ps, alphas, x_ref, prev_control_inputs):
        if self.num_tests == 0:
            print("No Monte Carlo tests to run.")
            return
        else:
            print(f"Running {self.num_tests} Monte Carlo tests...")

        for test_num in range(self.num_tests):
            xs, control_inputs = self.mp_dynamics.compute_op_point(Ps, alphas, x_ref, prev_control_inputs, 0.02 , test_num, False)

            test_cost = self._compute_cost(xs, control_inputs)
            self.total_costs.append(test_cost)
            self._check_collisions(xs)

        self._print_summary()

    def _compute_cost(self, xs, control_inputs):
        # get the linearized constraint matrices
        Gs, qs, rhos = self.mp_dynamics.get_Gs(xs, self.prox_cost_list, self.sigmas)

        # Iterate over timesteps
        cost_cur = 0

        for t in range(self.mp_dynamics.TIMESTEPS):
            concatenated_states = np.concatenate([state[t] for state in xs])
            for i, robot in enumerate(self.mp_dynamics.agent_list):
                cost_cur += self.costs[i][0].evaluate(concatenated_states, control_inputs[i][t], Gs[i][t][0], qs[i][t][0], rhos[i][t][0], self.lambdas[i][0][t], self.Is[i][0][t])
        return cost_cur

    def _check_collisions(self, xs):
        # Checks for collisions in a single test run
        positions = xs[:, :, :2]  # Extract (x, y) positions
        collision_detected = False
        for t in range(positions.shape[1]):
            pos_t = positions[:, t, :]
            for i in range(self.mp_dynamics.num_agents):
                for j in range(i + 1, self.mp_dynamics.num_agents):
                    distance = np.linalg.norm(pos_t[i] - pos_t[j])
                    if distance < self.collision_dist:
                        self.num_coll += 1
                        collision_detected = True  # Set the flag
                        break  # Break the innermost loop
                if collision_detected:
                    break  # Break the middle loop
            if collision_detected:
                break

    def _print_summary(self):
        # Normalize the cost by dividing by 1000
        self.total_costs = np.array(self.total_costs) / 1000
        # Prints a summary of collision results
        avg_cost = sum(self.total_costs) / self.num_tests
        std_cost = np.std(self.total_costs)
        print(f"\nAverage cost over {self.num_tests} tests: {avg_cost}")
        print(f"Standard deviation of cost: {std_cost}")
        print(f"Total collisions detected: {self.num_coll}")
        print(f"------------------------------------------------------------------------")

