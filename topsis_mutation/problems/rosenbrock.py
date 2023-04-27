from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

class Rosenbrock(FloatProblem):
    def __init__(self, number_of_variables: int = 2):
        super(Rosenbrock, self).__init__()
        self.number_of_variables = number_of_variables
        self.number_of_objectives = 1
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']

        self.lower_bound = [-5.0] * number_of_variables
        self.upper_bound = [10.0] * number_of_variables

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables

        result = sum([100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(self.number_of_variables - 1)])

        solution.objectives[0] = result

        return solution

    def get_name(self) -> str:
        return 'Rosenbrock'