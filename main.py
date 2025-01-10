import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class TruckAssignment:
    def __init__(
        self, chromosome: list[int], data: pd.DataFrame, truck_capacity: int
    ) -> None:
        self.chromosome = chromosome
        self.data = data
        self.truck_capacity = truck_capacity
        self.fitness = self.calc_fitness()

    def procreate(self, mate: "TruckAssignment") -> "TruckAssignment":
        new_chromosome = []
        for gene_1, gene_2 in zip(self.chromosome, mate.chromosome):
            probability = random.random()
            if probability < 0.7:
                new_chromosome.append(gene_1)
            elif probability < 0.9:
                new_chromosome.append(gene_2)
            else:
                new_chromosome.append(mutated_gene(0, 10))
        return TruckAssignment(new_chromosome, self.data, self.truck_capacity)

    def calc_fitness(self) -> float:
        total_weight = np.zeros(10)
        total_profit = 0
        penalty = 0

        for idx, truck in enumerate(self.chromosome):
            if truck > 0:
                weight = self.data.loc[idx, "Vikt"]
                profit = self.data.loc[idx, "Förtjänst"] + self.data.loc[idx, "Penalty"]
                total_weight[truck - 1] += weight
                if total_weight[truck - 1] > self.truck_capacity:
                    penalty += (total_weight[truck - 1] - self.truck_capacity) ** 2
                else:
                    total_profit += profit

        if any(total_weight > self.truck_capacity):
            return -1e6

        return total_profit - penalty

    def __str__(self) -> str:
        return "[ " + (" ".join(map(str, self.chromosome))) + " ]"


def mutated_gene(floor: int = 0, ceiling: int = 10) -> int:
    return random.randint(floor, ceiling)


def random_chromosome(
    length: int, data: pd.DataFrame, truck_capacity: int
) -> list[int]:
    chromosome = [0] * length
    total_weight = np.zeros(10)

    for idx in range(length):
        possible_trucks = [
            truck
            for truck in range(1, 11)
            if total_weight[truck - 1] + data.loc[idx, "Vikt"] <= truck_capacity
        ]
        if possible_trucks:
            chosen_truck = random.choice(possible_trucks)
            chromosome[idx] = chosen_truck
            total_weight[chosen_truck - 1] += data.loc[idx, "Vikt"]

    return chromosome


def genetic_algorithm(
    data: pd.DataFrame,
    truck_capacity: int,
    generations: int = 100,
    population_size: int = 50,
):
    population = [
        TruckAssignment(
            random_chromosome(len(data), data, truck_capacity), data, truck_capacity
        )
        for _ in range(population_size)
    ]
    best_solution = None

    for generation in range(generations):
        population.sort(key=lambda x: x.fitness, reverse=True)
        if best_solution is None or population[0].fitness > best_solution.fitness:
            best_solution = population[0]

        survivor_size = population_size // 10
        new_generation = population[:survivor_size]

        while len(new_generation) < population_size:
            parent1, parent2 = random.sample(population[:survivor_size], 2)
            child = parent1.procreate(parent2)
            new_generation.append(child)

        population = new_generation

        print(f"Generation {generation + 1}, Best Fitness: {best_solution.fitness}")

    return best_solution


def calculate_statistics(data: pd.DataFrame, solution: TruckAssignment):
    assigned_packages = data[np.array(solution.chromosome) > 0]
    unassigned_packages = data[np.array(solution.chromosome) == 0]

    # Vikt och förtjänst statistik
    assigned_weight = assigned_packages["Vikt"].sum()
    assigned_profit = assigned_packages["Förtjänst"].sum()

    unassigned_weight = unassigned_packages["Vikt"].sum()
    unassigned_profit = unassigned_packages["Förtjänst"].sum()

    print(f"Total förtjänst för levererade paket: {assigned_profit}")
    print(f"Antal paket kvar i lager: {len(unassigned_packages)}")
    print(f"Totalt vikt kvar i lager: {unassigned_weight}")
    print(f"Totalt förtjänst kvar i lager: {unassigned_profit}")

    # Histogram
    plt.hist(assigned_packages["Vikt"], bins=10, alpha=0.7, label="Vikt - Levererade")
    plt.hist(
        unassigned_packages["Vikt"], bins=10, alpha=0.7, label="Vikt - Kvar i lager"
    )
    plt.legend()
    plt.show()


def print_results(data: pd.DataFrame, solution: TruckAssignment, truck_capacity: int):
    # Skapa statistik
    total_weight = np.zeros(10)
    total_profit = np.zeros(10)
    delivered_packages = 0

    for idx, truck in enumerate(solution.chromosome):
        if truck > 0:
            weight = data.loc[idx, "Vikt"]
            profit = data.loc[idx, "Förtjänst"] + data.loc[idx, "Penalty"]
            total_weight[truck - 1] += weight
            total_profit[truck - 1] += profit
            delivered_packages += 1

    # Paket som ej levererades
    unassigned_packages = data[np.array(solution.chromosome) == 0]
    delayed_count = len(unassigned_packages)
    delayed_profit = unassigned_packages["Förtjänst"].sum()
    penalty_sum = unassigned_packages["Penalty"].sum()

    # Skriv ut detaljer per bil
    print("Optimering avslutas efter X generationer (upprepade små förbättringar).\n")
    for truck_id in range(10):
        print(
            f"Bil {truck_id + 1}: Vikt = {total_weight[truck_id]:.1f} kg, "
            f"Förtjänst = {int(total_profit[truck_id])} kr."
        )

    # Sammanställning av totala resultat
    print(f"\n{delayed_count} st försenade varor kvar.")
    print(f"Total daglig förtjänst: {int(total_profit.sum())} kr.")
    print(f"Total straffavgift på grund av förseningar: {int(penalty_sum)} kr.")
    actual_profit = int(total_profit.sum() + penalty_sum)
    print(f"Faktisk vinst: {actual_profit} kr.")
    print(f"Totalt levererade paket: {delivered_packages} paket.")


if __name__ == "__main__":
    # Läs in lagerstatus
    file_path = "lagerstatus.csv"
    data = pd.read_csv(file_path)
    data["Penalty"] = data["Deadline"].apply(lambda d: -((abs(d)) ** 2) if d < 0 else 0)

    TRUCK_CAPACITY = 800
    POPULATION_SIZE = 100
    GENERATIONS = 10

    # Kör genetisk algoritm
    best_solution = genetic_algorithm(
        data, TRUCK_CAPACITY, GENERATIONS, POPULATION_SIZE
    )

    # Skriv ut resultaten
    print_results(data, best_solution, TRUCK_CAPACITY)
