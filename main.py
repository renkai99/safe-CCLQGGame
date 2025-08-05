from configuration import Configuration

# Other solvers and scenes coming soon
solver = 'PD'
scene = 'lane_changing' # 'lane_changing' or 'intersection'

config = Configuration(scenario=scene)
algorithm = config.get_algorithm(solver)
xs, us, Ps, alphas = algorithm.run(config.xs, config.control_inputs, config.Ps, config.alphas, config.costs)

# Monte Carlo Test
num_monte_carlo_tests = 100
safety_tester = config.get_monte_carlo_tester(num_monte_carlo_tests)
safety_tester.run_tests(Ps, alphas, xs, us)

# Plot results
plotter = config.get_plotter(xs)
plotter.generate_output()