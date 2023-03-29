# Import necessary modules
from jmetal.algorithm.singleobjective import GeneticAlgorithm
from jmetal.operator import SBXCrossover, PolynomialMutation, BinaryTournamentSelection
from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.util.termination_criterion import StoppingByEvaluations
import matplotlib.pyplot as plt


"""
RASTRIGIN

The Rastrigin function is a well-known optimization test problem often used to evaluate
the performance of optimization algorithms. It is a non-linear, multi-modal function
characterized by its many local minima. The global minimum is located at the origin
(0, 0, ..., 0), where the function value is also 0.
"""
problem = Rastrigin(number_of_variables=10)



# Custom genetic algorithm with added fitness data storage for the plot
class CustomGeneticAlgorithm(GeneticAlgorithm):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.best_fitness_history = []

    def replacement(self, population, offspring_population):
        new_population = super().replacement(population, offspring_population)

        # Store the best fitness value every 100 evaluations
        if self.evaluations % 100 == 0:
            best_solution = min(new_population, key=lambda x: x.objectives[0])
            self.best_fitness_history.append(best_solution.objectives[0])

        return new_population
"""
Set up the Genetic Algorithm

- problem: the optimization problem
- population_size: the number of individuals in the population
- offspring_population_size: the number of offspring generated in each iteration
- mutation: the mutation operator
- crossover: the crossover operator
- selection: the selection operator for choosing parents for reproduction
- termination_criterion: the stopping condition for the algorithm
"""
algorithm = CustomGeneticAlgorithm(
    problem=problem,
    population_size=100,
    offspring_population_size=100,
    mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
    crossover=SBXCrossover(probability=0.9, distribution_index=20),
    selection=BinaryTournamentSelection(),
    termination_criterion=StoppingByEvaluations(max_evaluations=25000)
)

"""
MUTATION

Mutation is a genetic operator that introduces small random changes in the individual's genotype.
In the code, the Polynomial Mutation operator is used, which perturbs the variable values
according to a polynomial distribution. The mutation probability and distribution index
are its main parameters.


CROSSOVER

Crossover, also known as recombination, is a genetic operator that combines the genetic material
of two parent individuals to create offspring. In the code, the Simulated Binary Crossover (SBX)
operator is used, which simulates the behavior of one-point crossover in binary-coded
Genetic Algorithms but works with real-coded variables. The crossover probability
and distribution index are its main parameters.

"""

# Run the algorithm
algorithm.run()

# Retrieve and print the results
result = algorithm.get_result()
print("Solution: ", result.variables)
print("Fitness: ", result.objectives[0])

# Plot the fitness history
plt.plot(algorithm.best_fitness_history)
plt.title("Best Fitness Over Time")
plt.xlabel("Evaluations (x100)")
plt.ylabel("Best Fitness")
plt.grid()
plt.show()

# sposob 1 - w czasie dzialania
# sposob 2 - zamiast mutacji
# zamiast zupełnie losowej zmiany - zgodnie z TOPSIS