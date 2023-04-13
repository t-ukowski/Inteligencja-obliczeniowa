from jmetal.operator import PolynomialMutation
from jmetal.core.solution import FloatSolution

class TopsisMutation(PolynomialMutation):
    """
    TopsisMutation to klasa mutacji, która dziedziczy po PolynomialMutation.
    Implementuje mutację opartą na średniej wartości topowych osobników populacji.
    """
    def __init__(self, probability: float, selected_percentage: float, push_strength: float, best=True, worst=False, population=None):
        # Wywołanie konstruktora klasy bazowej
        super(TopsisMutation, self).__init__(probability=probability)
        self.selected_percentage = selected_percentage  # Procent najlepszych/najgorszych osobników do uśrednienia
        self.push_strength = push_strength    # Współczynnik określający siłę przyciągania do uśrednionego topowego osobnika
        self.population = population or []    # Populacja osobników
        self.best = best
        self.worst = worst


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

            # Przesunięcie wartości zmiennych mutowanego osobnika w kierunku uśrednionego osobnika
            for i in range(solution.number_of_variables):
                difference = average_best_individual[i] - solution.variables[i]
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

            # Przesunięcie wartości zmiennych mutowanego osobnika
            # w kierunku przeciwnym do uśrednionego osobnika
            for i in range(solution.number_of_variables):
                difference = solution.variables[i] - average_worst_individual[i]
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