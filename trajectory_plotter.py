# trajectory_plotter.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter

class TrajectoryPlotter:
    def __init__(self, xs_real, mp_dynamics, output_type='gif'):
        self.xs_real = xs_real
        self.mp_dynamics = mp_dynamics
        self.output_type = output_type
        self.x_traj = [[] for _ in range(mp_dynamics.num_agents)]
        self.y_traj = [[] for _ in range(mp_dynamics.num_agents)]
        self.headings = [[] for _ in range(mp_dynamics.num_agents)]
        self.width = 0.5
        self.height = 0.3
        self.colors = ['r', 'g', 'b', 'c', 'm', 'y']
        self.collect_trajectories()

    def collect_trajectories(self):
        for t in range(self.mp_dynamics.TIMESTEPS):
            for i, agent in enumerate(self.mp_dynamics.agent_list):
                self.x_traj[i].append(self.xs_real[i][t][0])
                self.y_traj[i].append(self.xs_real[i][t][1])
                self.headings[i].append(self.xs_real[i][t][2])

    def setup_plot(self):
        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.grid(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        return fig, ax

    def plot_robot(self, ax, x, y, heading, color):
        angle = np.degrees(heading)
        if heading > np.pi/2:
            rect = patches.Rectangle((x + self.width/2, y + self.height/2), self.width, self.height, angle=angle, color=color, fill=True)
        elif heading < -np.pi/8:
            rect = patches.Rectangle((x - self.width/2, y + self.height/2), self.width, self.height, angle=angle, color=color, fill=True)
        else:
            rect = patches.Rectangle((x - self.width/2, y - self.height/2), self.width, self.height, angle=angle, color=color, fill=True)
        ax.add_patch(rect)

        # Draw the heading arrow
        arrow_length = 0.5
        dx = arrow_length * np.cos(heading)
        dy = arrow_length * np.sin(heading)
        ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1, fc=color, ec=color)

    def update_frame(self, ax, kk, x_history, y_history):
        ax.clear()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        self.setup_plot_boundaries(ax)

        # Update and plot trajectories
        for i in range(self.mp_dynamics.num_agents):
            x_history[i].append(self.x_traj[i][kk])
            y_history[i].append(self.y_traj[i][kk])
            ax.plot(x_history[i], y_history[i], self.colors[i] + '--', linewidth=1)
            self.plot_robot(ax, self.x_traj[i][kk], self.y_traj[i][kk], self.headings[i][kk], self.colors[i])

        ax.text(-3.8, 3.8, f't = {kk*0.2:.1f}', fontsize=18, verticalalignment='top', horizontalalignment='left',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    def setup_plot_boundaries(self, ax):
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

    def generate_output(self):
        fig, ax = self.setup_plot()
        x_history = [[] for _ in range(self.mp_dynamics.num_agents)]
        y_history = [[] for _ in range(self.mp_dynamics.num_agents)]

        if self.output_type == 'gif':
            ani = FuncAnimation(fig, lambda kk: self.update_frame(ax, kk, x_history, y_history),
                                frames=self.mp_dynamics.TIMESTEPS, repeat=False)
            ani.save('./Plots/trajectory.gif', writer=PillowWriter(fps=10))
        elif self.output_type == 'eps':
            for kk in range(self.mp_dynamics.TIMESTEPS):
                self.update_frame(ax, kk, x_history, y_history)
                plt.savefig(f'./Plots/trajectory_f{kk:02d}.eps', format='eps')

        plt.ioff()
