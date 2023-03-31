import numpy as np
import matplotlib.pyplot as plt
from jmetal.algorithm.singleobjective import GeneticAlgorithm
from jmetal.operator import SBXCrossover, PolynomialMutation, BinaryTournamentSelection
from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.util.termination_criterion import StoppingByEvaluations
from .topsis_mutation import TopsisMutation
import json

class CustomGeneticAlgorithm(GeneticAlgorithm):

    def __init__(self, **kwargs):
        """
        Inicjalizacja obiektu CustomGeneticAlgorithm.

        Args:
            **kwargs: Argumenty przekazywane do konstruktora klasy nadrzędnej GeneticAlgorithm.
        """
        super().__init__(**kwargs)
        self.best_fitness_history = []  # Lista przechowująca historię najlepszych wartości przystosowania

    def replacement(self, population, offspring_population):
        """
        Metoda odpowiedzialna za zastępowanie starej populacji nową populacją potomków.

        Args:
            population: Stara populacja (lista obiektów FloatSolution).
            offspring_population: Nowa populacja potomków (lista obiektów FloatSolution).

        Returns:
            new_population: Aktualizowana populacja (lista obiektów FloatSolution).
        """
        # Wywołanie metody z klasy nadrzędnej
        new_population = super().replacement(population, offspring_population)

        # Rejestrowanie najlepszego rozwiązania co 100 ewaluacji
        if self.evaluations % 100 == 0:
            best_solution = min(new_population, key=lambda x: x.objectives[0])
            self.best_fitness_history.append(best_solution.objectives[0])
        
        # Aktualizacja populacji dla obiektu mutacji TopPercentageAveragingMutation
        if isinstance(self.mutation_operator, TopsisMutation):
            self.mutation_operator.population = new_population

        return new_population


def run_experiment(algorithm: CustomGeneticAlgorithm, max_evaluations: int):
    algorithm.run()

    best_solution = algorithm.get_result()
    print("Solution: ", best_solution.variables)
    print("Fitness: ", best_solution.objectives[0])

    history_length = max_evaluations // 100
    padded_history = np.zeros(history_length)
    padded_history[:len(algorithm.best_fitness_history)] = algorithm.best_fitness_history
    padded_history[len(algorithm.best_fitness_history):] = algorithm.best_fitness_history[-1]
    return padded_history


if __name__ == "__main__":

    # Load configuration from the JSON file
    with open("topsis_mutation/config.json", "r") as config_file:
        config = json.load(config_file)
    
    # Read parameters from the configuration
    num_tests = config["num_tests"]
    number_of_variables = config["number_of_variables"]
    ga_params = config["genetic_algorithm_params"]
    poly_mutation_params = config["polynomial_mutation_params"]
    topsis_mutation_params = config["topsis_mutation_params"]

    # Initialize the Rastrigin problem
    problem = Rastrigin(number_of_variables=number_of_variables)

    # Run the experiments
    poly_history_sum = np.zeros(int(ga_params["max_evaluations"]/100))
    top_percentage_history_sum = np.zeros(int(ga_params["max_evaluations"]/100))
    
    for _ in range(num_tests):
        # PolynomialMutation
        poly_algorithm = CustomGeneticAlgorithm(
            problem=problem,
            population_size=ga_params["population_size"],
            offspring_population_size=ga_params["offspring_population_size"],
            mutation=PolynomialMutation(
                probability=poly_mutation_params["probability"],
                distribution_index=poly_mutation_params["distribution_index"]
            ),
            crossover=SBXCrossover(probability=0.9, distribution_index=20),
            selection=BinaryTournamentSelection(),
            termination_criterion=StoppingByEvaluations(max_evaluations=ga_params["max_evaluations"])
        )
        poly_history = run_experiment(poly_algorithm, ga_params["max_evaluations"])
        poly_history_sum += poly_history
        
        # TopPercentageAveragingMutation
        top_percentage_algorithm = CustomGeneticAlgorithm(
            problem=problem,
            population_size=ga_params["population_size"],
            offspring_population_size=ga_params["offspring_population_size"],
            mutation=TopsisMutation(
                probability=topsis_mutation_params["probability"],
                top_percentage=topsis_mutation_params["top_percentage"],
                push_strength=topsis_mutation_params["push_strength"],
                population=[]
            ),
            crossover=SBXCrossover(probability=0.9, distribution_index=20),
            selection=BinaryTournamentSelection(),
            termination_criterion=StoppingByEvaluations(max_evaluations=ga_params["max_evaluations"])
        )
        top_percentage_history = run_experiment(top_percentage_algorithm, ga_params["max_evaluations"])
        
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
    plt.savefig("plots/topsis_mutation/combined_plot.png")
    plt.show()

    plt.plot(poly_history_avg)
    plt.title("Polynomial Mutation - Best Fitness Over Time")
    plt.xlabel("Evaluations (x100)")
    plt.ylabel("Best Fitness")
    plt.grid()
    plt.savefig("plots/topsis_mutation/polynomial_mutation_plot.png")
    # plt.show()

    plt.plot(top_percentage_history_avg)
    plt.title("Top Percentage Averaging Mutation - Best Fitness Over Time")
    plt.xlabel("Evaluations (x100)")
    plt.ylabel("Best Fitness")
    plt.grid()
    plt.savefig("plots/topsis_mutation/top_percentage_averaging_mutation_plot.png")
    # plt.show()