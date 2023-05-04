import math
import numpy as np
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

class ShiftedRotatedWeierstrass(FloatProblem):
    def __init__(self, number_of_variables: int = 2):
        super(ShiftedRotatedWeierstrass, self).__init__()
        self.number_of_variables = number_of_variables
        self.number_of_objectives = 1
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']

        self.lower_bound = [-0.5] * number_of_variables
        self.upper_bound = [0.5] * number_of_variables

        self.shift = np.random.uniform(-0.5, 0.5, size=self.number_of_variables)
        self.rotation_matrix = self.generate_random_orthogonal_matrix()

    def generate_random_orthogonal_matrix(self):
        random_matrix = np.random.randn(self.number_of_variables, self.number_of_variables)
        q, _ = np.linalg.qr(random_matrix)
        return q

    def weierstrass(self, x):
        k_max = 20
        a = 0.5
        b = 3

        sum1 = sum([a ** k * math.cos(2 * math.pi * b ** k * (xi + 0.5)) for k in range(k_max + 1) for xi in x])
        sum2 = sum([a ** k * math.cos(2 * math.pi * b ** k * 0.5) for k in range(k_max + 1)])

        return sum1 - self.number_of_variables * sum2

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = np.array(solution.variables)

        # Shift
        shifted_x = x - self.shift

        # Rotate
        rotated_x = np.dot(self.rotation_matrix, shifted_x)

        # Calculate Weierstrass function value
        result = self.weierstrass(rotated_x)

        solution.objectives[0] = result

        return solution

    def get_name(self) -> str:
        return 'Shifted Rotated Weierstrass'