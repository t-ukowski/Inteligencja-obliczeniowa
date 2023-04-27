import math
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

class Griewank(FloatProblem):
    def __init__(self, number_of_variables: int = 2):
        super(Griewank, self).__init__()
        self.number_of_variables = number_of_variables
        self.number_of_objectives = 1
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']

        self.lower_bound = [-600.0] * number_of_variables
        self.upper_bound = [600.0] * number_of_variables

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables

        sum_term = sum([xi**2 / 4000 for xi in x])
        prod_term = math.prod([math.cos(xi / math.sqrt(i + 1)) for i, xi in enumerate(x)])

        result = sum_term - prod_term + 1

        solution.objectives[0] = result

        return solution

    def get_name(self) -> str:
        return 'Griewank'