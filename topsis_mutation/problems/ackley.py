import math
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

class Ackley(FloatProblem):
    def __init__(self, number_of_variables: int = 2):
        super(Ackley, self).__init__()
        self.number_of_variables = number_of_variables
        self.number_of_objectives = 1
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']

        self.lower_bound = [-32.768] * number_of_variables
        self.upper_bound = [32.768] * number_of_variables

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables

        a = 20
        b = 0.2
        c = 2 * math.pi

        sum1 = sum([xi**2 for xi in x]) / self.number_of_variables
        sum2 = sum([math.cos(c * xi) for xi in x]) / self.number_of_variables
        result = -a * math.exp(-b * math.sqrt(sum1)) - math.exp(sum2) + a + math.e

        solution.objectives[0] = result

        return solution

    def get_name(self) -> str:
        return 'Ackley'