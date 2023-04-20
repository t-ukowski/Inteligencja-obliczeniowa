# Utils
import numpy as np
import matplotlib.pyplot as plt
# JMetalPy
from jmetal.algorithm.singleobjective import GeneticAlgorithm
from jmetal.operator import SBXCrossover, PolynomialMutation, BinaryTournamentSelection
from jmetal.util.termination_criterion import StoppingByEvaluations
# Mutations
from topsis_mutation.mutations.topsis_mutation import TopsisMutation
# Problems
from jmetal.problem.singleobjective.unconstrained import Rastrigin
from topsis_mutation.problems.ackley import Ackley
from topsis_mutation.problems.schwefel import Schwefel
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
    topsis1_params = config["topsis1_params"]
    topsis2_params = config["topsis2_params"]
    topsis3_params = config["topsis3_params"]

    # Initialize the problem
    problemType = config["problem"]
    match problemType:
        case "rastrigin":
            problem = Rastrigin(number_of_variables=number_of_variables)
        case "ackley":
            problem = Ackley(number_of_variables=number_of_variables)
        case "schwefel":
            problem = Schwefel(number_of_variables=number_of_variables)
        case _:
            problem = Rastrigin(number_of_variables=number_of_variables)

    # Run the experiments
    poly_history_sum = np.zeros(int(ga_params["max_evaluations"]/100))
    topsis1_history_sum = np.zeros(int(ga_params["max_evaluations"]/100))
    topsis2_history_sum = np.zeros(int(ga_params["max_evaluations"]/100))
    topsis3_history_sum = np.zeros(int(ga_params["max_evaluations"]/100))
    
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
        
        # TopsisMutation 1
        topsis1_algorithm = CustomGeneticAlgorithm(
            problem=problem,
            population_size=ga_params["population_size"],
            offspring_population_size=ga_params["offspring_population_size"],
            mutation=TopsisMutation(
                probability=topsis1_params["probability"],
                selected_percentage=topsis1_params["top_percentage"],
                push_strength=topsis1_params["push_strength"],
                best=topsis1_params["toBest"],
                worst=topsis1_params["fromWorst"],
                randomized_angle=topsis1_params["randomizedAngle"],
                randomized_point=topsis1_params["randomizedPoint"],
                population=[]
            ),
            crossover=SBXCrossover(probability=0.9, distribution_index=20),
            selection=BinaryTournamentSelection(),
            termination_criterion=StoppingByEvaluations(max_evaluations=ga_params["max_evaluations"])
        )
        topsis1_history = run_experiment(topsis1_algorithm, ga_params["max_evaluations"])
        
        # Set the population for the mutation object
        topsis1_algorithm.mutation_operator.set_mutation_population(topsis1_algorithm.solutions)
        
        topsis1_history_sum += topsis1_history


        # TopsisMutation 2
        topsis2_algorithm = CustomGeneticAlgorithm(
            problem=problem,
            population_size=ga_params["population_size"],
            offspring_population_size=ga_params["offspring_population_size"],
            mutation=TopsisMutation(
                probability=topsis2_params["probability"],
                selected_percentage=topsis2_params["top_percentage"],
                push_strength=topsis2_params["push_strength"],
                best=topsis2_params["toBest"],
                worst=topsis2_params["fromWorst"],
                randomized_angle=topsis2_params["randomizedAngle"],
                randomized_point=topsis2_params["randomizedPoint"],
                population=[]
            ),
            crossover=SBXCrossover(probability=0.9, distribution_index=20),
            selection=BinaryTournamentSelection(),
            termination_criterion=StoppingByEvaluations(max_evaluations=ga_params["max_evaluations"])
        )
        topsis2_history = run_experiment(topsis2_algorithm, ga_params["max_evaluations"])
        
        # Set the population for the mutation object
        topsis2_algorithm.mutation_operator.set_mutation_population(topsis2_algorithm.solutions)
        
        topsis2_history_sum += topsis2_history


        # TopsisMutation 3
        topsis3_algorithm = CustomGeneticAlgorithm(
            problem=problem,
            population_size=ga_params["population_size"],
            offspring_population_size=ga_params["offspring_population_size"],
            mutation=TopsisMutation(
                probability=topsis3_params["probability"],
                selected_percentage=topsis3_params["top_percentage"],
                push_strength=topsis3_params["push_strength"],
                best=topsis3_params["toBest"],
                worst=topsis3_params["fromWorst"],
                randomized_angle=topsis3_params["randomizedAngle"],
                randomized_point=topsis3_params["randomizedPoint"],
                population=[]
            ),
            crossover=SBXCrossover(probability=0.9, distribution_index=20),
            selection=BinaryTournamentSelection(),
            termination_criterion=StoppingByEvaluations(max_evaluations=ga_params["max_evaluations"])
        )
        topsis3_history = run_experiment(topsis3_algorithm, ga_params["max_evaluations"])
        
        # Set the population for the mutation object
        topsis3_algorithm.mutation_operator.set_mutation_population(topsis3_algorithm.solutions)
        
        topsis3_history_sum += topsis3_history


    # Calculate average histories
    poly_history_avg = poly_history_sum / num_tests
    topsis1_history_avg = topsis1_history_sum / num_tests
    topsis2_history_avg = topsis2_history_sum / num_tests
    topsis3_history_avg = topsis3_history_sum / num_tests

    # All experiments comparison
    plt.plot(poly_history_avg, label="Polynomial Mutation")
    plt.plot(topsis1_history_avg, label="Topsis Mutation (<params1>)")
    plt.plot(topsis2_history_avg, label="Topsis Mutation (<params2>)")
    plt.plot(topsis3_history_avg, label="Topsis Mutation (<params3>)")
    plt.title("Best Fitness Over Time")
    plt.xlabel("Evaluations (<number>)")
    plt.ylabel("Best Fitness")
    plt.legend()
    plt.grid()
    plt.savefig("plots/topsis_mutation/all_combined_plot.png")
    plt.show()

    # Three experiments comparison
    plt.plot(topsis1_history_avg, label="Topsis Mutation (<params1>)")
    plt.plot(topsis2_history_avg, label="Topsis Mutation (<params2>)")
    plt.plot(topsis3_history_avg, label="Topsis Mutation (<params3>)")
    plt.title("Best Fitness Over Time")
    plt.xlabel("Evaluations (<number>)")
    plt.ylabel("Best Fitness")
    plt.legend()
    plt.grid()
    plt.savefig("plots/topsis_mutation/three_combined_plot.png")
    # plt.show()


    # Two experiments comparison
    plt.plot(poly_history_avg, label="Polynomial Mutation")
    plt.plot(topsis1_history_avg, label="Topsis Mutation")
    plt.title("Best Fitness Over Time")
    plt.xlabel("Evaluations (x100)")
    plt.ylabel("Best Fitness")
    plt.legend()
    plt.grid()
    plt.savefig("plots/topsis_mutation/two_combined_plot.png")
    # plt.show()

    plt.plot(poly_history_avg)
    plt.title("Polynomial Mutation - Best Fitness Over Time")
    plt.xlabel("Evaluations (x100)")
    plt.ylabel("Best Fitness")
    plt.grid()
    plt.savefig("plots/topsis_mutation/polynomial_mutation_plot.png")
    # plt.show()

    plt.plot(topsis1_history_avg)
    plt.title("Topsis Mutation - Best Fitness Over Time")
    plt.xlabel("Evaluations (x100)")
    plt.ylabel("Best Fitness")
    plt.grid()
    plt.savefig("plots/topsis_mutation/top_percentage_averaging_mutation_plot.png")
    # plt.show()
