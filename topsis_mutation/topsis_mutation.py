import math
import random
from jmetal.operator import PolynomialMutation
from jmetal.core.solution import FloatSolution

def scaled_random_vector_in_angle_range(angle_range_degrees: float, dimensions: int, scale_factor: float) -> list:
    random_angles = [random.uniform(-angle_range_degrees / 2, angle_range_degrees / 2) for _ in range(dimensions)]
    random_vectors = [scale_factor * math.cos(math.radians(angle)) for angle in random_angles]
    return random_vectors

def random_point_in_bounding_box(lower_bound, upper_bound, dimensions):
    return [random.uniform(lower_bound[i], upper_bound[i]) for i in range(dimensions)]

class TopsisMutation(PolynomialMutation):
    """
    TopsisMutation to klasa mutacji, która dziedziczy po PolynomialMutation.
    Implementuje mutację opartą na średniej wartości topowych osobników populacji.
    """
    def __init__(self, probability: float, selected_percentage: float, push_strength: float, best=True, worst=False, randomized_angle=False, randomized_point=False, population=None):
        # Wywołanie konstruktora klasy bazowej
        super(TopsisMutation, self).__init__(probability=probability)
        self.selected_percentage = selected_percentage  # Procent najlepszych/najgorszych osobników do uśrednienia
        self.push_strength = push_strength    # Współczynnik określający siłę przyciągania do uśrednionego topowego osobnika
        self.population = population or []    # Populacja osobników
        self.best = best
        self.worst = worst
        self.randomized_angle = randomized_angle
        self.randomized_point = randomized_point


    def execute(self, solution: FloatSolution) -> FloatSolution:
        """
        Funkcja wykonująca mutację.
        """
        # Posortowanie populacji według wartości funkcji celu
        sorted_population = sorted(self.population, key=lambda x: x.objectives[0])

        # Obliczenie liczby osobników do uśrednienia
        num_individuals = int(len(sorted_population) * self.selected_percentage)
        
        # Zapewniamy, że zostanie wybrany co najmniej jeden osobnik
        num_individuals = max(num_individuals, 1)

        # Wybranie najlepszych osobników
        top_individuals = sorted_population[:num_individuals]
        # Wybranie najgorszych osobników
        bottom_individuals = sorted_population[num_individuals:]

        if self.best:
            # Inicjalizacja listy przechowującej uśrednionego najlepszego osobnika
            average_best_individual = [0.0] * solution.number_of_variables

            # Sumowanie wartości zmiennych topowych osobników
            for individual in top_individuals:
                for i in range(solution.number_of_variables):
                    average_best_individual[i] += individual.variables[i]
            # Obliczenie wartości uśrednionego topowego osobnika
            for i in range(solution.number_of_variables):
                average_best_individual[i] /= num_individuals
            
            if self.randomized_point:
                random_point = random_point_in_bounding_box(solution.variables, average_best_individual, solution.number_of_variables)

            # Przesunięcie wartości zmiennych mutowanego osobnika w kierunku uśrednionego osobnika
            for i in range(solution.number_of_variables):
                difference = average_best_individual[i] - solution.variables[i]

                # Przyciągamy się do losowego punktu w kwadracie między obecnym osobnikiem a uśrednionym docelowym
                if self.randomized_point:
                    difference = random_point[i] - solution.variables[i]
                # Dodanie losowości dla kierunku wektora, jeśli flaga jest aktywna
                if self.randomized_angle:
                    scale_factor = abs(difference) / 2
                    random_vectors = scaled_random_vector_in_angle_range(90, 1, scale_factor)
                    difference += random_vectors[0]

                solution.variables[i] += self.push_strength * difference

                # Sprawdzenie, czy nowa wartość zmiennej nie przekracza granic
                if solution.variables[i] < solution.lower_bound[i]:
                    solution.variables[i] = solution.lower_bound[i]
                if solution.variables[i] > solution.upper_bound[i]:
                    solution.variables[i] = solution.upper_bound[i]

        if self.worst:
            # Inicjalizacja listy przechowującej uśrednionego najgorszego osobnika
            average_worst_individual = [0.0] * solution.number_of_variables

            # Sumowanie wartości zmiennych najgorszych osobników
            for individual in bottom_individuals:
                for i in range(solution.number_of_variables):
                    average_worst_individual[i] += individual.variables[i]
            # Obliczenie wartości uśrednionego najgorszego osobnika
            for i in range(solution.number_of_variables):
                average_worst_individual[i] /= num_individuals

            if self.randomized_point:
                random_point = random_point_in_bounding_box(solution.variables, average_worst_individual, solution.number_of_variables)

            # Przesunięcie wartości zmiennych mutowanego osobnika
            # w kierunku przeciwnym do uśrednionego osobnika
            for i in range(solution.number_of_variables):
                difference = solution.variables[i] - average_worst_individual[i]

                # Przyciągamy się do losowego punktu w kwadracie między obecnym osobnikiem a uśrednionym docelowym
                if self.randomized_point:
                    difference = solution.variables[i] - random_point[i]
                # Dodanie losowości dla kierunku wektora, jeśli flaga jest aktywna
                if self.randomized_angle:
                    scale_factor = abs(difference) / 2
                    random_vectors = scaled_random_vector_in_angle_range(90, 1, scale_factor)
                    difference += random_vectors[0]

                solution.variables[i] += self.push_strength * difference

                # Sprawdzenie, czy nowa wartość zmiennej nie przekracza granic
                if solution.variables[i] < solution.lower_bound[i]:
                    solution.variables[i] = solution.lower_bound[i]
                if solution.variables[i] > solution.upper_bound[i]:
                    solution.variables[i] = solution.upper_bound[i]

        return solution

    def set_mutation_population(self, population):
        """
        Metoda do aktualizacji populacji dla obiektu mutacji.
        """
        self.population = population

    def get_name(self):
        return 'Top Percentage Averaging Mutation'