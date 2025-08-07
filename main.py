from configuration import Configuration

# Other solvers and scenes coming soon
solver = 'PD'
scene = 'intersection' # 'lane_changing' or 'intersection'

# Two modes of reference trajectory: 
# (1) known: with a given reference trajectory (as in our paper): https://ieeexplore.ieee.org/document/11077446 
# (2) iterative: iterative LQ game (based on D.F.Keil 2020): https://ieeexplore.ieee.org/document/9197129
ref_traj_type = 'iterative'  # 'knwon' or 'iterative'

config = Configuration(scenario=scene, ref_traj_type=ref_traj_type)
algorithm = config.get_algorithm(solver)
xs, us, Ps, alphas = algorithm.run(config.xs, config.control_inputs, config.Ps, config.alphas, config.costs)

# Monte Carlo Test
num_monte_carlo_tests = 100
safety_tester = config.get_monte_carlo_tester(num_monte_carlo_tests)
safety_tester.run_tests(Ps, alphas, xs, us)

# Plot results
plotter = config.get_plotter(xs)
plotter.generate_output()