import math
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

class Schwefel(FloatProblem):
    def __init__(self, number_of_variables: int = 2):
        super(Schwefel, self).__init__()
        self.number_of_variables = number_of_variables
        self.number_of_objectives = 1
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']

        self.lower_bound = [-500.0] * number_of_variables
        self.upper_bound = [500.0] * number_of_variables

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables
        result = 0

        for xi in x:
            result += xi * math.sin(math.sqrt(abs(xi)))

        result = 418.9829 * self.number_of_variables - result
        solution.objectives[0] = result

        return solution

    def get_name(self) -> str:
        return 'Schwefel'