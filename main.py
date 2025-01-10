import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Konstanter
MAX_WEIGHT, NUM_TRUCKS, POP_SIZE, NUM_ITER, MUT_RATE, ELITISM = 800, 10, 20, 100, 0.1, 2
WEIGHT_PENALTY, DEADLINE_PRIORITY = 0.01, 1.5


# Läs data
def read_data(file_path: Path) -> np.ndarray:
    with file_path.open("r", encoding="utf-8") as file:
        return np.array(
            [
                [
                    int(row["Paket_id"]),
                    float(row["Vikt"]),
                    int(row["Förtjänst"]),
                    int(row["Deadline"]),
                ]
                for row in csv.DictReader(file)
            ]
        )


# Tilldelning av paket
def allocate_packages(packages: np.ndarray):
    packages = packages[np.lexsort((-packages[:, 2], packages[:, 3]))]
    trucks = [np.empty((0, packages.shape[1])) for _ in range(NUM_TRUCKS)]
    truck_weights = np.zeros(NUM_TRUCKS)
    for package in packages:
        for truck_id in range(NUM_TRUCKS):
            if truck_weights[truck_id] + package[1] <= MAX_WEIGHT:
                trucks[truck_id] = np.vstack([trucks[truck_id], package])
                truck_weights[truck_id] += package[1]
                break
    return trucks, packages[~np.isin(packages[:, 0], np.vstack(trucks)[:, 0])]


# Beräkning av fitness
def calculate_fitness(trucks, remaining):
    delayed_packages = remaining[remaining[:, 3] < 0]
    profit = sum(np.sum(truck[:, 2]) for truck in trucks if truck.size)
    penalty = -np.sum(delayed_packages[:, 3] ** 2) if delayed_packages.size else 0
    unused = sum(
        WEIGHT_PENALTY * (MAX_WEIGHT - np.sum(truck[:, 1]))
        for truck in trucks
        if truck.size
    )
    return profit + penalty - unused


# Initial population
def init_population(packages):
    return [np.random.permutation(packages) for _ in range(POP_SIZE)]


# Mutation
def mutate(solution, rate):
    for _ in range(max(1, int(len(solution) * rate))):
        idx = np.random.choice(len(solution), 2, replace=False)
        solution[idx[0]], solution[idx[1]] = solution[idx[1]], solution[idx[0]]
    return solution


# Statistik
def show_statistics(trucks, remaining):
    weights, profits = [np.sum(truck[:, 1]) for truck in trucks if truck.size], [
        np.sum(truck[:, 2]) for truck in trucks if truck.size
    ]
    plt.figure(figsize=(10, 6))
    plt.bar(
        range(1, NUM_TRUCKS + 1), weights, label="Vikt (kg)", alpha=0.7, color="blue"
    )
    plt.bar(
        range(1, NUM_TRUCKS + 1),
        profits,
        label="Förtjänst (kr)",
        alpha=0.7,
        color="green",
        bottom=weights,
    )
    plt.xlabel("Budbilar")
    plt.ylabel("Värde")
    plt.title("Vikt och Förtjänst per Budbil")
    plt.legend()
    plt.tight_layout()
    plt.show()


def present_results(trucks, remaining):
    print("\nRESULTATREDOVISNING:")
    total_profit = 0
    total_weight = 0
    for i, truck in enumerate(trucks):
        if truck.size:
            truck_weight = np.sum(truck[:, 1])
            truck_profit = np.sum(truck[:, 2])
            total_weight += truck_weight
            total_profit += truck_profit
            print(
                f"Bil {i + 1}: Vikt = {truck_weight:.1f} kg, Förtjänst = {truck_profit:.1f} kr."
            )
    remaining_count = len(remaining)
    remaining_weight = np.sum(remaining[:, 1]) if remaining.size else 0
    total_penalty = (
        -np.sum(remaining[remaining[:, 3] < 0, 3] ** 2) if remaining.size else 0
    )
    actual_profit = total_profit + total_penalty
    print(f"\n{remaining_count} st försenade varor kvar.")
    print(f"Total daglig förtjänst: {total_profit:.1f} kr.")
    print(f"Total straffavgift på grund av förseningar: {total_penalty:.1f} kr.")
    print(f"Faktisk vinst: {actual_profit:.1f} kr.")
    print(f"Totalt levererade paket: {sum(len(truck) for truck in trucks)} paket.")


# Optimering
def optimize(packages):
    population, best_fitness, tracker = init_population(packages), float("-inf"), []
    for iteration in range(NUM_ITER):
        fitness_population = []
        for solution in population:
            trucks, remaining = allocate_packages(solution)
            fitness = calculate_fitness(trucks, remaining)
            fitness_population.append((fitness, solution))
            if fitness > best_fitness:
                best_fitness, best_solution = fitness, (trucks, remaining)
        tracker.append(best_fitness)
        print(f"Iteration {iteration + 1}: Best Fitness = {best_fitness}")
        fitness_population.sort(reverse=True, key=lambda x: x[0])
        next_gen = [sol for _, sol in fitness_population[:ELITISM]]
        while len(next_gen) < POP_SIZE:
            parent = random.choice(fitness_population[: POP_SIZE // 2])[1]
            next_gen.append(mutate(parent.copy(), MUT_RATE))
        population = next_gen
        if len(tracker) > 10 and all(
            abs(tracker[-1] - tracker[-i]) < 0.0001 for i in range(1, 11)
        ):
            print(
                "Optimeringen avstannar efter {iteration + 1} generationer (upprepade små förbättringar)."
            )
            break
    show_statistics(*best_solution)
    present_results(*best_solution)
    return tracker


# Main
if __name__ == "__main__":
    file_path = Path("lagerstatus.csv")
    packages = read_data(file_path)
    optimize(packages)
