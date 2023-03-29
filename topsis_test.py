import sys
import numpy as np
import matplotlib.pyplot as plt
from jmetal.algorithm.singleobjective import GeneticAlgorithm
from jmetal.operator import SBXCrossover, PolynomialMutation, BinaryTournamentSelection
from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.util.termination_criterion import StoppingByEvaluations
from typing import List
from jmetal.core.solution import FloatSolution
from jmetal.core.operator import Mutation


class TopPercentageAveragingMutation(PolynomialMutation):

    def __init__(self, probability: float, top_percentage: float, push_strength: float, population=None):
        super(TopPercentageAveragingMutation, self).__init__(probability=probability)
        self.top_percentage = top_percentage
        self.push_strength = push_strength
        self.population = population or []

    def execute(self, solution: FloatSolution) -> FloatSolution:
        sorted_population = sorted(self.population, key=lambda x: x.objectives[0])
        num_top_individuals = int(len(sorted_population) * self.top_percentage)
        # Ensure at least one individual is selected
        num_top_individuals = max(num_top_individuals, 1)
        top_individuals = sorted_population[:num_top_individuals]
        average_individual = [0.0] * solution.number_of_variables
        for individual in top_individuals:
            for i in range(solution.number_of_variables):
                average_individual[i] += individual.variables[i]
        for i in range(solution.number_of_variables):
            average_individual[i] /= num_top_individuals

        for i in range(solution.number_of_variables):
            difference = average_individual[i] - solution.variables[i]
            solution.variables[i] += self.push_strength * difference

            if solution.variables[i] < solution.lower_bound[i]:
                solution.variables[i] = solution.lower_bound[i]
            if solution.variables[i] > solution.upper_bound[i]:
                solution.variables[i] = solution.upper_bound[i]

        return solution

    def set_mutation_population(self, population):
        self.population = population

    def get_name(self):
        return 'Top Percentage Averaging Mutation'
    

class CustomGeneticAlgorithm(GeneticAlgorithm):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.best_fitness_history = []

    def replacement(self, population, offspring_population):
        new_population = super().replacement(population, offspring_population)

        if self.evaluations % 100 == 0:
            best_solution = min(new_population, key=lambda x: x.objectives[0])
            self.best_fitness_history.append(best_solution.objectives[0])
        
        # Update the mutation population
        if isinstance(self.mutation_operator, TopPercentageAveragingMutation):
            self.mutation_operator.population = new_population

        return new_population


def run_experiment(algorithm: CustomGeneticAlgorithm, max_evaluations: int):
    algorithm.run()

    best_solution = algorithm.get_result()
    print("Solution: ", best_solution.variables)
    print("Fitness: ", best_solution.objectives[0])

    history = np.zeros(max_evaluations // 100)
    history[:len(algorithm.best_fitness_history)] = algorithm.best_fitness_history
    return history


if __name__ == "__main__":
    # Read the number of tests to run from the console
    num_tests = int(input("Enter the number of tests to run: "))

    # Initialize the Rastrigin problem
    problem = Rastrigin(number_of_variables=10)

    # Run the experiments
    poly_history_sum = np.zeros(250)
    top_percentage_history_sum = np.zeros(250)
    
    for _ in range(num_tests):
        # PolynomialMutation
        poly_algorithm = CustomGeneticAlgorithm(
            problem=problem,
            population_size=100,
            offspring_population_size=100,
            mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
            crossover=SBXCrossover(probability=0.9, distribution_index=20),
            selection=BinaryTournamentSelection(),
            termination_criterion=StoppingByEvaluations(max_evaluations=25000)
        )
        poly_history = run_experiment(poly_algorithm, 25000)
        poly_history_sum += poly_history
        
        # TopPercentageAveragingMutation
        top_percentage_algorithm = CustomGeneticAlgorithm(
            problem=problem,
            population_size=100,
            offspring_population_size=100,
            mutation=TopPercentageAveragingMutation(probability=1.0 / problem.number_of_variables, top_percentage=0.1, push_strength=0.1, population=[]),
            crossover=SBXCrossover(probability=0.9, distribution_index=20),
            selection=BinaryTournamentSelection(),
            termination_criterion=StoppingByEvaluations(max_evaluations=25000)
        )
        top_percentage_history = run_experiment(top_percentage_algorithm, 25000)
        
        # Set the population for the mutation object
        top_percentage_algorithm.mutation_operator.set_mutation_population(top_percentage_algorithm.solutions)

        top_percentage_history_sum += top_percentage_history

    # Calculate average histories
    poly_history_avg = poly_history_sum / num_tests
    top_percentage_history_avg = top_percentage_history_sum / num_tests


    plt.plot(poly_history_avg, label="Polynomial Mutation")
    plt.plot(top_percentage_history_avg, label="Top Percentage Averaging Mutation")
    plt.title("Best Fitness Over Time")
    plt.xlabel("Evaluations (x100)")
    plt.ylabel("Best Fitness")
    plt.legend()
    plt.grid()
    plt.savefig("combined_plot.png")
    plt.show()

    plt.plot(poly_history_avg)
    plt.title("Polynomial Mutation - Best Fitness Over Time")
    plt.xlabel("Evaluations (x100)")
    plt.ylabel("Best Fitness")
    plt.grid()
    plt.savefig("polynomial_mutation_plot.png")
    plt.show()

    plt.plot(top_percentage_history_avg)
    plt.title("Top Percentage Averaging Mutation - Best Fitness Over Time")
    plt.xlabel("Evaluations (x100)")
    plt.ylabel("Best Fitness")
    plt.grid()
    plt.savefig("top_percentage_averaging_mutation_plot.png")
    plt.show()