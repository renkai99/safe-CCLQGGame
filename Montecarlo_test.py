# monte_carlo_safety_test.py
import numpy as np
from Costs import ProximityCost
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt


class MonteCarloTest:
    def __init__(self, scenario, mp_dynamics, num_tests, collision_dist, costs, sigmas, lambdas, Is):
        self.scenario = scenario
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

    def setup_plot(self):
        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0, right=0.1, top=0.1, bottom=0)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.grid(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        return fig, ax

    def setup_plot_boundaries(self, ax):
        if self.scenario == 'intersection':
            road_width = 2
            road_limits1, road_limits2 = [-4, -2], [2, 4]
            ax.fill_between(road_limits1, road_width, 4, color='gray', alpha=0.5)
            ax.fill_between(road_limits1, -4, -road_width, color='gray', alpha=0.5)
            ax.fill_between(road_limits2, road_width, 4, color='gray', alpha=0.5)
            ax.fill_between(road_limits2, -4, -road_width, color='gray', alpha=0.5)

            # Road boundaries
            ax.plot([-4, -road_width], [road_width, road_width], 'k', linewidth=2)
            ax.plot([-4, -road_width], [-road_width, -road_width], 'k', linewidth=2)
            ax.plot([road_width, 4], [road_width, road_width], 'k', linewidth=2)
            ax.plot([road_width, 4], [-road_width, -road_width], 'k', linewidth=2)
            ax.plot([road_width, road_width], [4, road_width], 'k', linewidth=2)
            ax.plot([-road_width, -road_width], [4, road_width], 'k', linewidth=2)
            ax.plot([road_width, road_width], [-4, -road_width], 'k', linewidth=2)
            ax.plot([-road_width, -road_width], [-4, -road_width], 'k', linewidth=2)
        elif self.scenario == 'lane_changing':
            road_width = 3
            ax.fill_between([-4, 4], road_width, 4, color='gray', alpha=0.5)
            ax.fill_between([-4, 4], -4, -road_width, color='gray', alpha=0.5)
            ax.plot([-4, 4], [-road_width, -road_width], color='k', linewidth=5, alpha=0.5)
            ax.plot([-4, 4], [road_width, road_width], color='k', linewidth=5, alpha=0.5)


            # Dash lines for lanes with lower frequency and more white space
            for i in range(-1, 3, 2):
                ax.plot([-4, 4], [i, i], linestyle=(0, (5, 10)), color='k', linewidth=5, alpha=0.5)
        else:
            print("Scenario not recognized. No boundaries plotted.")

    def run_tests(self, Ps, alphas, x_ref, prev_control_inputs):
        if self.num_tests == 0:
            print("No Monte Carlo tests to run.")
            return
        else:
            print(f"Running {self.num_tests} Monte Carlo tests...")

        xs_history = []  # Initialize a list to store all xs

        for test_num in range(self.num_tests):
            xs, control_inputs = self.mp_dynamics.compute_op_point(Ps, alphas, x_ref, prev_control_inputs, 0.02, test_num, False)
            
            xs_history.append(xs)  # Store the xs for this test
            test_cost = self._compute_cost(xs, control_inputs)
            self.total_costs.append(test_cost)
            self._check_collisions(xs)

        self._print_summary()
        self._plot_trajectories(xs_history)

    def _plot_car(self, ax, start_point, color, orientation):
        # Define car dimensions
        car_length = 0.5
        car_width = 0.3
        wheel_radius = 0.06

        # Create a rectangle to represent the car body
        if orientation == 'vertical':
            car_body = Rectangle(
                (start_point[0] - car_width / 2, start_point[1] - car_length / 2),
                car_width,
                car_length,
                color=color,
                alpha=0.8
            )
            # car_body.angle = -90  # Rotate the car body 90 degrees clockwise
        else:
            car_body = Rectangle(
                (start_point[0] - car_length / 2, start_point[1] - car_width / 2),
                car_length,
                car_width,
                color=color,
                alpha=0.8
            )
        ax.add_patch(car_body)

        # Add wheels to the car
        if orientation == 'vertical':  # Adjust wheel positions for the rotated car
            wheel_positions = [
                (start_point[0] - car_width / 2 - wheel_radius, start_point[1] - car_length / 2 + wheel_radius),  # Front-left
                (start_point[0] - car_width / 2 - wheel_radius, start_point[1] + car_length / 2 - wheel_radius),  # Front-right
                (start_point[0] + car_width / 2 + wheel_radius, start_point[1] - car_length / 2 + wheel_radius),  # Rear-left
                (start_point[0] + car_width / 2 + wheel_radius, start_point[1] + car_length / 2 - wheel_radius)   # Rear-right
            ]
        else:
            wheel_positions = [
                (start_point[0] - car_length / 2 + wheel_radius, start_point[1] - car_width / 2 - wheel_radius),  # Front-left
                (start_point[0] + car_length / 2 - wheel_radius, start_point[1] - car_width / 2 - wheel_radius),  # Front-right
                (start_point[0] - car_length / 2 + wheel_radius, start_point[1] + car_width / 2 + wheel_radius),  # Rear-left
                (start_point[0] + car_length / 2 - wheel_radius, start_point[1] + car_width / 2 + wheel_radius)   # Rear-right
            ]

        for wheel_pos in wheel_positions:
            wheel = plt.Circle(wheel_pos, wheel_radius, color='k', alpha=0.9)
            ax.add_patch(wheel)

        # Add windows to the car
        if orientation == 'vertical':  # Adjust window positions for the rotated car
            window_width = car_width * 0.4
            window_height = car_length * 0.4
            window_positions = [
                (start_point[0] - window_width / 2, start_point[1] - window_height / 2)  # Center window
            ]
        else:
            window_width = car_length * 0.4
            window_height = car_width * 0.4
            window_positions = [
                (start_point[0] - window_width / 2, start_point[1] - window_height / 2)  # Center window
            ]

        for window_pos in window_positions:
            window = Rectangle(
                window_pos,
                window_width,
                window_height,
                color='skyblue',
                alpha=0.6
            )
            ax.add_patch(window)

    def _plot_trajectories(self, xs_history):
        fig, ax = self.setup_plot()
        ax.clear()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        self.setup_plot_boundaries(ax)
        colors = ['r', 'b', 'g', 'y', 'c', 'm']  # Define a list of colors for players
        for agent_idx in range(xs_history[0].shape[0]):  # Iterate over agents
            # Draw trajectory curves for each test
            for xs in xs_history:
                # Generate a perturbed color for each agent
                base_color = np.array(plt.cm.colors.to_rgb(colors[agent_idx % len(colors)]))
                perturbation = np.random.uniform(-0.2, 0.2, size=3)  # Random perturbation within a small range
                perturbed_color = np.clip(base_color + perturbation, 0, 1)  # Ensure RGB values remain valid
                waypoints = np.concatenate([xs[agent_idx, :, :2]], axis=0)  # Collect all waypoints
                ax.scatter(waypoints[:, 0], waypoints[:, 1], color=perturbed_color, alpha=0.05)
                trajectory = xs[agent_idx, :, :2]  # Extract trajectory for the agent
                ax.plot(trajectory[:, 0], trajectory[:, 1], color=perturbed_color, alpha=0.3)

                # Draw a car at the starting point of the trajectory
                start_point = xs_history[0][agent_idx, 0, :2]  # Get the starting point of the first trajectory
                # Plot cars for three agents
                if agent_idx == 2:  # Third agent
                    self._plot_car(ax, start_point, colors[agent_idx % len(colors)], orientation='vertical')
                else:
                    self._plot_car(ax, start_point, colors[agent_idx % len(colors)], orientation='horizontal')

                

        # Add a single legend for each agent with the base color
        for agent_idx in range(xs_history[0].shape[0]):
            ax.plot([], [], color=colors[agent_idx % len(colors)], label=f"Agent {agent_idx+1}")

        # Add a magnified inset

        # Create the inset axes
        axins = zoomed_inset_axes(ax, zoom=5, loc='lower left', borderpad = 0.1)  # Zoom factor and location
        axins.set_xlim(-1, -0.5)  # Set x-limits for the inset
        axins.set_ylim(1.05, 1.52)  # Set y-limits for the inset
        axins.set_xticks([])  # Remove x-ticks
        axins.set_yticks([])  # Remove y-ticks

        # Plot the same content as the main plot in the inset
        for agent_idx in range(xs_history[0].shape[0]):
            for xs in xs_history:
                base_color = np.array(plt.cm.colors.to_rgb(colors[agent_idx % len(colors)]))
                perturbation = np.random.uniform(-0.4, 0.4, size=3)
                perturbed_color = np.clip(base_color + perturbation, 0, 1)
                waypoints = np.concatenate([xs[agent_idx, :, :2]], axis=0)
                axins.scatter(waypoints[:, 0], waypoints[:, 1], color=perturbed_color, alpha=0.3, s=10)
                trajectory = xs[agent_idx, :, :2]
                axins.plot(trajectory[:, 0], trajectory[:, 1], color=perturbed_color, alpha=0.7, linewidth=0.1)

        # Mark the inset region on the main plot
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

        ax.legend(loc='upper right', fontsize=14)
        plt.show()

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

