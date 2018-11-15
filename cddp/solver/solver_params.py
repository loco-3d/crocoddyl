import yaml

class SolverParams(object):
  def __init__(self):
    """ Define the default values of the DDP parameters
    """
    # Defining the default values of the DDP parameters
    # Convergence tolerance and maximum number of iterations
    self.tol = 1e-5
    self.max_iter = 20

    # Resizing the global variables for analysing solver performance
    self.cost_itr = [0.] * self.max_iter
    self.muLM_itr = [0.] * self.max_iter
    self.muV_itr = [0.] * self.max_iter
    self.gamma_itr = [0.] * self.max_iter
    self.theta_itr = [0.] * self.max_iter
    self.alpha_itr = [0.] * self.max_iter

    # Regularization parameters (factor and increased rate)
    self.mu0LM = 0.
    self.muLM_inc = 10.
    self.muLM_dec = 0.5
    self.mu0V = 0.
    self.muV_inc = 10.
    self.muV_dec = 0.5

    # Line search parameters (step, lower and upper bound of iteration
    # acceptance and decreased rate)
    self.alpha0 = 1.
    self.alpha_min = 1e-3
    self.alpha_inc = 2.
    self.alpha_dec = 0.5
    self.armijo_condition = 1e-3
    self.change_ub = 100.
    return

  def setFromConfigFile(self, config_file):
    """ Sets the properties of the DDP solver from a YAML file.

    :param config_file: YAML configuration file
    """
    # Reading the YAML file
    with open(config_file, 'r') as stream:
      data = yaml.load(stream)
      # Setting up stop criteria
      self.tol = float(data['ddp']['stop_criteria']['tol'])
      self.max_iter = int(data['ddp']['stop_criteria']['max_iter'])

      # Resizing the global variables for analysing solver performance
      self.cost_itr = [0.] * self.max_iter
      self.muLM_itr = [0.] * self.max_iter
      self.muV_itr = [0.] * self.max_iter
      self.gamma_itr = [0.] * self.max_iter
      self.theta_itr = [0.] * self.max_iter
      self.alpha_itr = [0.] * self.max_iter

      # Setting up regularization
      self.mu0LM = float(data['ddp']['regularization']['levenberg_marquard']['mu0'])
      self.muLM_inc = float(data['ddp']['regularization']['levenberg_marquard']['inc_rate'])
      self.muLM_dec = float(data['ddp']['regularization']['levenberg_marquard']['dec_rate'])
      self.mu0V = float(data['ddp']['regularization']['value_function']['mu0'])
      self.muV_inc = float(data['ddp']['regularization']['value_function']['inc_rate'])
      self.muV_dec = float(data['ddp']['regularization']['value_function']['dec_rate'])

      # Setting up line search
      self.alpha0 = float(data['ddp']['line_search']['alpha0'])
      self.alpha_min = float(data['ddp']['line_search']['min_stepsize'])
      self.alpha_inc = float(data['ddp']['line_search']['inc_rate'])
      self.alpha_dec = float(data['ddp']['line_search']['dec_rate'])
      self.armijo_condition = float(data['ddp']['line_search']['armijo_condition'])
      self.change_ub = float(data['ddp']['line_search']['change_ub'])
