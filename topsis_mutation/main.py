# Window
import tkinter as tk
from tkinter import ttk
from matplotlib import patches
# Utils
import numpy as np
import matplotlib.pyplot as plt
# JMetalPy
from jmetal.algorithm.singleobjective import GeneticAlgorithm
from jmetal.operator import SBXCrossover, PolynomialMutation, BinaryTournamentSelection, SimpleRandomMutation, UniformMutation
from jmetal.util.termination_criterion import StoppingByEvaluations
# Mutations
from topsis_mutation.mutations.topsis_mutation import TopsisMutation
# Problems
from jmetal.problem.singleobjective.unconstrained import Rastrigin
from topsis_mutation.problems.ackley import Ackley
from topsis_mutation.problems.schwefel import Schwefel
from topsis_mutation.problems.rosenbrock import Rosenbrock
from topsis_mutation.problems.griewank import Griewank
from topsis_mutation.problems.weierstrass import ShiftedRotatedWeierstrass

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


class Application(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title("GA Configuration")
        self.geometry("1015x600")

        # General options
        general_frame = ttk.LabelFrame(self, text="General Options")
        general_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Number of tests
        ttk.Label(general_frame, text="Number of Tests:").grid(row=0, column=0)
        self.num_tests = ttk.Entry(general_frame)
        self.num_tests.insert(0, "7")
        self.num_tests.grid(row=0, column=1)

        # Problem type
        ttk.Label(general_frame, text="Problem Type:").grid(row=1, column=0)
        self.problem_type = tk.StringVar()
        self.problem_type.set("Rastrigin")
        ttk.Radiobutton(general_frame, text="Rastrigin", variable=self.problem_type, value="Rastrigin").grid(row=1, column=1, padx=5, pady=5)
        ttk.Radiobutton(general_frame, text="Ackley", variable=self.problem_type, value="Ackley").grid(row=1, column=2, padx=5, pady=5)
        ttk.Radiobutton(general_frame, text="Schwefel", variable=self.problem_type, value="Schwefel").grid(row=1, column=3, padx=5, pady=5)
        ttk.Radiobutton(general_frame, text="Rosenbrock", variable=self.problem_type, value="Rosenbrock").grid(row=1, column=4, padx=5, pady=5)
        ttk.Radiobutton(general_frame, text="Griewank", variable=self.problem_type, value="Griewank").grid(row=1, column=5, padx=5, pady=5)
        ttk.Radiobutton(general_frame, text="Shifted Rotated Weierstrass", variable=self.problem_type, value="Shifted Rotated Weierstrass").grid(row=1, column=6, padx=5, pady=5)

        # Number of variables
        ttk.Label(general_frame, text="Number of Variables:").grid(row=2, column=0, padx=5, pady=5)
        self.number_of_variables = ttk.Entry(general_frame)
        self.number_of_variables.insert(0, "100")
        self.number_of_variables.grid(row=2, column=1)

        # Genetic Algorithm Params
        ttk.Label(general_frame, text="Population Size:").grid(row=3, column=0, padx=5, pady=5)
        self.population_size = ttk.Entry(general_frame)
        self.population_size.insert(0, "100")
        self.population_size.grid(row=3, column=1)

        ttk.Label(general_frame, text="Offspring Population Size:").grid(row=4, column=0, padx=5, pady=5)
        self.offspring_population_size = ttk.Entry(general_frame)
        self.offspring_population_size.insert(0, "100")
        self.offspring_population_size.grid(row=4, column=1)

        ttk.Label(general_frame, text="Max Evaluations:").grid(row=5, column=0, padx=5, pady=5)
        self.max_evaluations = ttk.Entry(general_frame)
        self.max_evaluations.insert(0, "25000")
        self.max_evaluations.grid(row=5, column=1)

        # Mutation options
        mutation_frame = ttk.LabelFrame(self, text="Mutation Options")
        mutation_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # SimpleRandom mutation params
        self.simple_mutation_vars = {
            "enabled": tk.BooleanVar(value=True), # Change "True" > "Flase"
        }

        self.simple_mutation_frame = ttk.LabelFrame(mutation_frame, text="Simple Random Mutation")
        self.simple_mutation_frame.grid(row=0, column=0, padx=5, pady=5)

        ttk.Label(self.simple_mutation_frame, text="Probability").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.simple_mutation_probability = ttk.Entry(self.simple_mutation_frame)
        self.simple_mutation_probability.insert(0, "0.01")
        self.simple_mutation_probability.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # The on/off button
        self.simple_on_off_button = ttk.Checkbutton(self.simple_mutation_frame, text="Enable", variable=self.simple_mutation_vars["enabled"], command=self.toggle_simple_mutation_frame)
        self.simple_on_off_button.grid(row=0, column=2, sticky="w")

        # Polynomial mutation params
        self.poly_mutation_vars = {
            "enabled": tk.BooleanVar(value=True), # Change "True" > "Flase"
        }

        self.poly_mutation_frame = ttk.LabelFrame(mutation_frame, text="Polynomial Mutation")
        self.poly_mutation_frame.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(self.poly_mutation_frame, text="Probability").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.poly_mutation_probability = ttk.Entry(self.poly_mutation_frame)
        self.poly_mutation_probability.insert(0, "0.01")
        self.poly_mutation_probability.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(self.poly_mutation_frame, text="Distribution Index").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.poly_mutation_distribution = ttk.Entry(self.poly_mutation_frame)
        self.poly_mutation_distribution.insert(0, "20")
        self.poly_mutation_distribution.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # The on/off button
        self.poly_on_off_button = ttk.Checkbutton(self.poly_mutation_frame, text="Enable", variable=self.poly_mutation_vars["enabled"], command=self.toggle_poly_mutation_frame)
        self.poly_on_off_button.grid(row=0, column=2, sticky="w")

        # Uniform mutation params
        self.uniform_mutation_vars = {
            "enabled": tk.BooleanVar(value=True), # Change "True" > "Flase"
        }

        self.uniform_mutation_frame = ttk.LabelFrame(mutation_frame, text="Uniform Mutation")
        self.uniform_mutation_frame.grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(self.uniform_mutation_frame, text="Probability").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.uniform_mutation_probability = ttk.Entry(self.uniform_mutation_frame)
        self.uniform_mutation_probability.insert(0, "0.01")
        self.uniform_mutation_probability.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(self.uniform_mutation_frame, text="Perturbation").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.uniform_mutation_perturbation = ttk.Entry(self.uniform_mutation_frame)
        self.uniform_mutation_perturbation.insert(0, "0.3")
        self.uniform_mutation_perturbation.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # The on/off button
        self.uniform_on_off_button = ttk.Checkbutton(self.uniform_mutation_frame, text="Enable", variable=self.uniform_mutation_vars["enabled"], command=self.toggle_uniform_mutation_frame)
        self.uniform_on_off_button.grid(row=0, column=2, sticky="w")

        # Topsis mutation params
        self.topsis_mutation_vars = []
        for i in range(3):
            self.topsis_mutation_vars.append({
                "probability": tk.DoubleVar(value=0.4),
                "top_percentage": tk.DoubleVar(value=0.4),
                "push_strength": tk.DoubleVar(value=0.4),
                "toBest": tk.BooleanVar(),
                "fromWorst": tk.BooleanVar(),
                "randomizedAngle": tk.BooleanVar(),
                "randomizedPoint": tk.BooleanVar(),
                "enabled": tk.BooleanVar(value=True),
            })
        self.topsis_mutation_vars[0]["toBest"].set(True)
        self.topsis_mutation_vars[0]["fromWorst"].set(True)
        
        self.topsis_mutation_vars[1]["probability"].set(0.1)
        self.topsis_mutation_vars[1]["top_percentage"].set(0.3)
        self.topsis_mutation_vars[1]["push_strength"].set(0.6)
        self.topsis_mutation_vars[1]["toBest"].set(True)
        self.topsis_mutation_vars[1]["fromWorst"].set(True)
        self.topsis_mutation_vars[1]["randomizedAngle"].set(True)


        self.topsis_mutation_vars[2]["probability"].set(0.07)
        self.topsis_mutation_vars[2]["top_percentage"].set(0.25)
        self.topsis_mutation_vars[2]["push_strength"].set(0.4)
        self.topsis_mutation_vars[2]["toBest"].set(True)
        self.topsis_mutation_vars[2]["fromWorst"].set(True)
        self.topsis_mutation_vars[2]["randomizedAngle"].set(True)


        for i in range(3):
            topsis_mutation_frame = ttk.LabelFrame(mutation_frame, text=f"Topsis Mutation {i+1}")
            topsis_mutation_frame.grid(row=1, column=i, padx=5)

            # The on/off button
            on_off_button = ttk.Checkbutton(topsis_mutation_frame, text="Enable", variable=self.topsis_mutation_vars[i]["enabled"], command=lambda idx=i: self.toggle_mutation_frame(idx))
            on_off_button.grid(row=0, column=2, sticky="w")
            self.topsis_mutation_vars[i]["on_off_button"] = {"widget": on_off_button}

            ttk.Label(topsis_mutation_frame, text="Probability").grid(row=0, column=0, sticky="w")
            prob_entry = ttk.Entry(topsis_mutation_frame, textvariable=self.topsis_mutation_vars[i]["probability"])
            prob_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")
            self.topsis_mutation_vars[i]["probability"].widget = prob_entry

            ttk.Label(topsis_mutation_frame, text="Top Percentage").grid(row=1, column=0, sticky="w")
            perc_entry = ttk.Entry(topsis_mutation_frame, textvariable=self.topsis_mutation_vars[i]["top_percentage"])
            perc_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")
            self.topsis_mutation_vars[i]["top_percentage"].widget = perc_entry

            ttk.Label(topsis_mutation_frame, text="Push Strength").grid(row=2, column=0, sticky="w")
            push_entry = ttk.Entry(topsis_mutation_frame, textvariable=self.topsis_mutation_vars[i]["push_strength"])
            push_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")
            self.topsis_mutation_vars[i]["push_strength"].widget = push_entry

            checkbutton1 = ttk.Checkbutton(topsis_mutation_frame, text="To Best", variable=self.topsis_mutation_vars[i]["toBest"])
            checkbutton1.grid(row=3, column=0, sticky="w")
            self.topsis_mutation_vars[i]["toBest"].widget = checkbutton1
            checkbutton2 = ttk.Checkbutton(topsis_mutation_frame, text="From Worst", variable=self.topsis_mutation_vars[i]["fromWorst"])
            checkbutton2.grid(row=3, column=1, sticky="w")
            self.topsis_mutation_vars[i]["fromWorst"].widget = checkbutton2
            checkbutton3 = ttk.Checkbutton(topsis_mutation_frame, text="Randomized Angle", variable=self.topsis_mutation_vars[i]["randomizedAngle"])
            checkbutton3.grid(row=4, column=0, sticky="w")
            self.topsis_mutation_vars[i]["randomizedAngle"].widget = checkbutton3
            checkbutton4 = ttk.Checkbutton(topsis_mutation_frame, text="Randomized Point", variable=self.topsis_mutation_vars[i]["randomizedPoint"])
            checkbutton4.grid(row=4, column=1, sticky="w")
            self.topsis_mutation_vars[i]["randomizedPoint"].widget = checkbutton4

        # Buttons
        buttons_frame = ttk.Frame(self)
        buttons_frame.grid(row=3, column=0, columnspan=3, pady=10, sticky="nsew")

        # Plot frame
        plot_frame = ttk.LabelFrame(self, text="Plotting Options")
        plot_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

        # Plot type
        ttk.Label(plot_frame, text="Plot Type:").grid(row=1, column=0)
        self.plot_type = tk.StringVar()
        self.plot_type.set("line")
        ttk.Radiobutton(plot_frame, text="Line plot", variable=self.plot_type, value="line").grid(row=1, column=1, padx=5, pady=5)
        ttk.Radiobutton(plot_frame, text="Box plot", variable=self.plot_type, value="box").grid(row=1, column=2, padx=5, pady=5)

        exit_button = ttk.Button(buttons_frame, text='Exit', command=self.destroy)
        exit_button.grid(row=0, column=0, padx=20)

        run_tests_button = ttk.Button(buttons_frame, text='Run Tests', command=self.run_tests)
        run_tests_button.grid(row=0, column=1, padx=800)


    def toggle_simple_mutation_frame(self):
        enabled = self.simple_mutation_vars["enabled"].get()

        for child in self.simple_mutation_frame.winfo_children():
            if (isinstance(child, ttk.Entry) or isinstance(child, ttk.Checkbutton) or isinstance(child, ttk.Label)) and child != self.simple_on_off_button:
                if enabled:
                    child.configure(state="normal")
                else:
                    child.configure(state="disabled")

    def toggle_poly_mutation_frame(self):
        enabled = self.poly_mutation_vars["enabled"].get()

        for child in self.poly_mutation_frame.winfo_children():
            if (isinstance(child, ttk.Entry) or isinstance(child, ttk.Checkbutton) or isinstance(child, ttk.Label)) and child != self.poly_on_off_button:
                if enabled:
                    child.configure(state="normal")
                else:
                    child.configure(state="disabled")

    def toggle_uniform_mutation_frame(self):
        enabled = self.uniform_mutation_vars["enabled"].get()

        for child in self.uniform_mutation_frame.winfo_children():
            if (isinstance(child, ttk.Entry) or isinstance(child, ttk.Checkbutton) or isinstance(child, ttk.Label)) and child != self.uniform_on_off_button:
                if enabled:
                    child.configure(state="normal")
                else:
                    child.configure(state="disabled")

    def toggle_mutation_frame(self, index):
        frame = self.topsis_mutation_vars[index]
        enabled = frame["enabled"].get()

        frame_widget = frame["probability"].widget.master
        for child in frame_widget.winfo_children():
            if (isinstance(child, ttk.Entry) or isinstance(child, ttk.Checkbutton) or isinstance(child, ttk.Label)) and child != frame["on_off_button"]["widget"]:
                if enabled:
                    child.configure(state="normal")
                else:
                    child.configure(state="disabled")

    def generate_box_data(self, history_table, num_boxes=30, sampling_number=250):
        section_size = int(sampling_number / num_boxes)
        box_data = [[] for _ in range(num_boxes)]

        for history in history_table:
            for i in range(num_boxes):
                start = i * section_size
                end = start + section_size
                section_avg = np.mean(history[start:end])
                box_data[i].append(section_avg)

        return box_data

    def run_tests(self):
        # Read the parameters
        num_tests = int(self.num_tests.get())
        number_of_variables = int(self.number_of_variables.get())
        population_size = int(self.population_size.get())
        offspring_population_size = int(self.offspring_population_size.get())
        max_evaluations = int(self.max_evaluations.get())

        simple_enabled = self.simple_mutation_vars["enabled"].get()
        if simple_enabled:
            simple_mutation_probability = float(self.simple_mutation_probability.get())

        poly_enabled = self.poly_mutation_vars["enabled"].get()
        if poly_enabled:
            poly_mutation_probability = float(self.poly_mutation_probability.get())
            poly_mutation_distribution = float(self.poly_mutation_distribution.get())
        
        uniform_enabled = self.uniform_mutation_vars["enabled"].get()
        if uniform_enabled:
            uniform_mutation_probability = float(self.uniform_mutation_probability.get())
            uniform_mutation_perturbation = float(self.uniform_mutation_perturbation.get())

        topsis_mutation_params = []
        for mutation_var in self.topsis_mutation_vars:
            params = {}
            for key, value in mutation_var.items():
                if isinstance(value, tk.Variable):
                    params[key] = value.get()
                elif isinstance(value, dict) and "widget" in value:
                    pass  # Ignore the widget reference
                else:
                    params[key] = value
            topsis_mutation_params.append(params)

        # Initialize the problem
        problem_type = self.problem_type.get()
        match problem_type:
            case "Rastrigin":
                problem = Rastrigin(number_of_variables=number_of_variables)
            case "Ackley":
                problem = Ackley(number_of_variables=number_of_variables)
            case "Schwefel":
                problem = Schwefel(number_of_variables=number_of_variables)
            case "Rosenbrock":
                problem = Rosenbrock(number_of_variables=number_of_variables)
            case "Griewank":
                problem = Griewank(number_of_variables=number_of_variables)
            case "Shifted Rotated Weierstrass":
                problem = ShiftedRotatedWeierstrass(number_of_variables=number_of_variables)
            case _:
                problem = Rastrigin(number_of_variables=number_of_variables)

        # Run the experiments
        sampling_rate = 100
        sampling_number = max_evaluations // sampling_rate


        history_keys = ['simple', 'poly', 'uniform', 'topsis1', 'topsis2', 'topsis3']

        history_sum = {key: np.zeros(sampling_number) for key in history_keys}
        history_table = {key: [] for key in history_keys}

        for _ in range(num_tests):
            # SimpleRandomMutation
            if simple_enabled:
                simple_algorithm = CustomGeneticAlgorithm(
                    problem=problem,
                    population_size=population_size,
                    offspring_population_size=offspring_population_size,
                    mutation=SimpleRandomMutation(
                        probability=simple_mutation_probability,
                    ),
                    crossover=SBXCrossover(probability=0.9, distribution_index=20),
                    selection=BinaryTournamentSelection(),
                    termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
                )
                simple_history = run_experiment(simple_algorithm, max_evaluations)
                history_sum['simple'] += simple_history
                history_table['simple'].append(simple_history)

            # PolynomialMutation
            if poly_enabled:
                poly_algorithm = CustomGeneticAlgorithm(
                    problem=problem,
                    population_size=population_size,
                    offspring_population_size=offspring_population_size,
                    mutation=PolynomialMutation(
                        probability=poly_mutation_probability,
                        distribution_index=poly_mutation_distribution
                    ),
                    crossover=SBXCrossover(probability=0.9, distribution_index=20),
                    selection=BinaryTournamentSelection(),
                    termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
                )
                poly_history = run_experiment(poly_algorithm, max_evaluations)
                history_sum['poly'] += poly_history
                history_table['poly'].append(poly_history)

            # UniformMutation
            if uniform_enabled:
                uniform_algorithm = CustomGeneticAlgorithm(
                    problem=problem,
                    population_size=population_size,
                    offspring_population_size=offspring_population_size,
                    mutation=UniformMutation(
                        probability=uniform_mutation_probability,
                        perturbation=uniform_mutation_perturbation
                    ),
                    crossover=SBXCrossover(probability=0.9, distribution_index=20),
                    selection=BinaryTournamentSelection(),
                    termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
                )
                uniform_history = run_experiment(uniform_algorithm, max_evaluations)
                history_sum['uniform'] += uniform_history
                history_table['uniform'].append(uniform_history)
            
            # TopsisMutation 1
            topsis1_params = topsis_mutation_params[0]
            if topsis1_params["enabled"]:
                topsis1_algorithm = CustomGeneticAlgorithm(
                    problem=problem,
                    population_size=population_size,
                    offspring_population_size=offspring_population_size,
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
                    termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
                )
                topsis1_history = run_experiment(topsis1_algorithm, max_evaluations)
                
                # Set the population for the mutation object
                topsis1_algorithm.mutation_operator.set_mutation_population(topsis1_algorithm.solutions)
                
                history_sum['topsis1'] += topsis1_history
                history_table['topsis1'].append(topsis1_history)


            # TopsisMutation 2
            topsis2_params = topsis_mutation_params[1]
            if topsis2_params["enabled"]:
                topsis2_algorithm = CustomGeneticAlgorithm(
                    problem=problem,
                    population_size=population_size,
                    offspring_population_size=offspring_population_size,
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
                    termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
                )
                topsis2_history = run_experiment(topsis2_algorithm, max_evaluations)
                
                # Set the population for the mutation object
                topsis2_algorithm.mutation_operator.set_mutation_population(topsis2_algorithm.solutions)
                
                history_sum['topsis2'] += topsis2_history
                history_table['topsis2'].append(topsis2_history)


            # TopsisMutation 3
            topsis3_params = topsis_mutation_params[2]
            if topsis3_params["enabled"]:
                topsis3_algorithm = CustomGeneticAlgorithm(
                    problem=problem,
                    population_size=population_size,
                    offspring_population_size=offspring_population_size,
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
                    termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
                )
                topsis3_history = run_experiment(topsis3_algorithm, max_evaluations)
                
                # Set the population for the mutation object
                topsis3_algorithm.mutation_operator.set_mutation_population(topsis3_algorithm.solutions)
                
                history_sum['topsis3'] += topsis3_history
                history_table['topsis3'].append(topsis3_history)


        # Calculate average histories
        if simple_enabled:
            simple_history_avg = history_sum['simple'] / num_tests
        if poly_enabled:
            poly_history_avg = history_sum['poly'] / num_tests
        if uniform_enabled:
            uniform_history_avg = history_sum['uniform'] / num_tests
        if topsis1_params["enabled"]:
            topsis1_history_avg = history_sum['topsis1'] / num_tests
        if topsis2_params["enabled"]:
            topsis2_history_avg = history_sum['topsis2'] / num_tests
        if topsis3_params["enabled"]:
            topsis3_history_avg = history_sum['topsis3'] / num_tests

        # All experiments comparison
        plot_type = self.plot_type.get()
        match plot_type:
            case "line":
                if simple_enabled:
                    plt.plot(simple_history_avg, label="Simple Random Mutation")
                if poly_enabled:
                    plt.plot(poly_history_avg, label="Polynomial Mutation")
                if uniform_enabled:
                    plt.plot(uniform_history_avg, label="Uniform Mutation")
                if topsis1_params["enabled"]:
                    selected_vars = {k: v for k, v in topsis1_params.items() if k in ["toBest", "fromWorst", "randomizedAngle", "randomizedPoint"]}
                    options_str = ", ".join(opt for opt, value in selected_vars.items() if value)
                    plt.plot(topsis1_history_avg, label=f"Topsis Mutation ({options_str})")
                if topsis2_params["enabled"]:
                    selected_vars = {k: v for k, v in topsis2_params.items() if k in ["toBest", "fromWorst", "randomizedAngle", "randomizedPoint"]}
                    options_str = ", ".join(opt for opt, value in selected_vars.items() if value)
                    plt.plot(topsis2_history_avg, label=f"Topsis Mutation ({options_str})")
                if topsis3_params["enabled"]:
                    selected_vars = {k: v for k, v in topsis3_params.items() if k in ["toBest", "fromWorst", "randomizedAngle", "randomizedPoint"]}
                    options_str = ", ".join(opt for opt, value in selected_vars.items() if value)
                    plt.plot(topsis3_history_avg, label=f"Topsis Mutation ({options_str})")
                plt.title(f"Best Fitness Over Time - {problem_type}")
                plt.xlabel(f"Evaluations ({max_evaluations})")
                plt.ylabel("Best Fitness")
                plt.legend()
                plt.grid()
                plt.savefig("plots/topsis_mutation/line_plot.png")
                plt.show()
            case "box":
                # Filter enabled mutation types
                enabled_mutation_types = [(key, history_table[key]) for key in history_keys if len(history_table[key]) > 0]

                num_boxes = 30

                # Generate box data for each enabled mutation type
                box_data_list = [self.generate_box_data(history, num_boxes, sampling_number) for _, history in enabled_mutation_types]

                # Calculate box positions for side-by-side box plots
                box_width = 0.8 / len(enabled_mutation_types)
                box_positions = [
                    [i + j * box_width for i in range(1, num_boxes + 1)]
                    for j in range(len(enabled_mutation_types))
                ]

                legend_patches = []

                # Plot the custom box plots side by side
                for i, (mutation_type, _) in enumerate(enabled_mutation_types):
                    plt.boxplot(box_data_list[i], positions=box_positions[i], widths=box_width, manage_ticks=False, patch_artist=True, boxprops=dict(facecolor="C" + str(i)))

                    # Add selected parameters to the legend
                    selected_vars = {k: v for k, v in eval(mutation_type + '_params').items() if k in ["toBest", "fromWorst", "randomizedAngle", "randomizedPoint"]}
                    options_str = ", ".join(opt for opt, value in selected_vars.items() if value)
                    # legend_label = f"{mutation_type} ({options_str})" # TODO: code below shows Topsis Mutation even if polynomial mutation is selected
                    legend_label = f"Topsis Mutation ({options_str})"
                    legend_patches.append(patches.Patch(color="C" + str(i), label=legend_label))

                # Customize x-axis ticks and labels
                ax = plt.gca()
                ax.set_xticks(range(1, num_boxes + 1))
                ax.set_xticklabels(range(1, num_boxes + 1))

                # Add the legend to the plot
                plt.legend(handles=legend_patches)

                # Plot the custom box plot
                plt.title(f"Best Fitness Over Time - {problem_type}")
                plt.xlabel("Section")
                plt.ylabel("Fitness")
                plt.grid(axis="y")
                plt.savefig("plots/topsis_mutation/box_plot.png")
                plt.show()

            case _:
                print("Invalid plot type provided")

if __name__ == "__main__":
   
    app = Application()
    app.title('Configuration')
    app.mainloop()
