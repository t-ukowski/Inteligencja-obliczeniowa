from jmetal.operator import PolynomialMutation
from jmetal.core.solution import FloatSolution

class TopPercentageAveragingMutation(PolynomialMutation):
    """
    TopPercentageAveragingMutation to klasa mutacji, która dziedziczy po PolynomialMutation.
    Implementuje mutację opartą na średniej wartości topowych osobników populacji.
    """
    def __init__(self, probability: float, top_percentage: float, push_strength: float, population=None):
        # Wywołanie konstruktora klasy bazowej
        super(TopPercentageAveragingMutation, self).__init__(probability=probability)
        self.top_percentage = top_percentage  # Procent najlepszych osobników do uśrednienia
        self.push_strength = push_strength    # Współczynnik określający siłę przyciągania do uśrednionego topowego osobnika
        self.population = population or []    # Populacja osobników

    def execute(self, solution: FloatSolution) -> FloatSolution:
        """
        Funkcja wykonująca mutację.
        """
        # Posortowanie populacji według wartości funkcji celu
        sorted_population = sorted(self.population, key=lambda x: x.objectives[0])

        # Obliczenie liczby najlepszych osobników do uśrednienia
        num_top_individuals = int(len(sorted_population) * self.top_percentage)
        
        # Zapewniamy, że zostanie wybrany co najmniej jeden osobnik
        num_top_individuals = max(num_top_individuals, 1)

        # Wybranie topowych osobników
        top_individuals = sorted_population[:num_top_individuals]

        # Inicjalizacja listy przechowującej uśrednionego osobnika
        average_individual = [0.0] * solution.number_of_variables

        # Sumowanie wartości zmiennych topowych osobników
        for individual in top_individuals:
            for i in range(solution.number_of_variables):
                average_individual[i] += individual.variables[i]
        # Obliczenie wartości uśrednionego topowego osobnika
        for i in range(solution.number_of_variables):
            average_individual[i] /= num_top_individuals

        # Przesunięcie wartości zmiennych mutowanego osobnika w kierunku uśrednionego osobnika
        for i in range(solution.number_of_variables):
            difference = average_individual[i] - solution.variables[i]
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