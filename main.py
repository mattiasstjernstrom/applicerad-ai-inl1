import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

MAX_WEIGHT = 800
NUM_TRUCKS = 10
NO_IMPROVEMENT_LIMIT = 30
POPULATION_SIZE = 20
NUM_GENERATIONS = 1000
MUTATION_RATE = 0.2


def read_data(file_path: Path) -> list:
    with file_path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return [
            {
                "Paket_id": int(row["Paket_id"]),
                "Vikt": float(row["Vikt"]),
                "Förtjänst": int(row["Förtjänst"]),
                "Deadline": int(row["Deadline"]),
            }
            for row in reader
        ]


def allocate_packages(packages: list) -> dict:
    packages = sorted(
        packages,
        key=lambda x: (x["Deadline"] <= 0, -x["Förtjänst"], x["Deadline"]),
    )
    trucks = defaultdict(list)
    truck_weights = np.zeros(NUM_TRUCKS)

    for package in packages:
        for truck_id in range(NUM_TRUCKS):
            if truck_weights[truck_id] + package["Vikt"] <= MAX_WEIGHT:
                trucks[truck_id].append(package)
                truck_weights[truck_id] += package["Vikt"]
                break

    return trucks


def calculate_fitness(trucks, remaining_packages):
    total_profit = np.sum([p["Förtjänst"] for truck in trucks.values() for p in truck])
    total_penalty = np.sum(
        [-(p["Deadline"] ** 2) for p in remaining_packages if p["Deadline"] < 0]
    )
    return total_profit + total_penalty


def initialize_population(packages):
    population = []
    for _ in range(POPULATION_SIZE):
        random.shuffle(packages)
        trucks = allocate_packages(packages)
        remaining_packages = [p for p in packages if p not in sum(trucks.values(), [])]
        fitness = calculate_fitness(trucks, remaining_packages)
        population.append((fitness, trucks, remaining_packages))
    return population


def mutate_solution(trucks, remaining_packages):
    truck_weights = np.array(
        [sum(p["Vikt"] for p in trucks[truck_id]) for truck_id in range(NUM_TRUCKS)]
    )

    for _ in range(int(len(remaining_packages) * MUTATION_RATE)):
        if random.random() < 0.5 and remaining_packages:
            package = remaining_packages.pop(
                random.randint(0, len(remaining_packages) - 1)
            )
            for truck_id in range(NUM_TRUCKS):
                if truck_weights[truck_id] + package["Vikt"] <= MAX_WEIGHT:
                    trucks[truck_id].append(package)
                    truck_weights[truck_id] += package["Vikt"]
                    break
            else:
                remaining_packages.append(package)
        else:
            truck_id = random.choice(list(trucks.keys()))
            if trucks[truck_id]:
                package = trucks[truck_id].pop(
                    random.randint(0, len(trucks[truck_id]) - 1)
                )
                truck_weights[truck_id] -= package["Vikt"]
                remaining_packages.append(package)

    return trucks, remaining_packages


def select_parents(population):
    return sorted(population, key=lambda x: x[0], reverse=True)[:5]


def crossover_solution(parent1, parent2):
    trucks1, remaining1 = parent1[1], parent1[2]
    trucks2, remaining2 = parent2[1], parent2[2]
    new_trucks = defaultdict(list)

    for truck_id in range(NUM_TRUCKS):
        if random.random() < 0.5:
            new_trucks[truck_id] = trucks1[truck_id]
        else:
            new_trucks[truck_id] = trucks2[truck_id]

    new_remaining = {p["Paket_id"]: p for p in remaining1 + remaining2}.values()
    return new_trucks, list(new_remaining)


def create_next_generation(parents):
    next_generation = []
    while len(next_generation) < POPULATION_SIZE:
        parent1, parent2 = random.sample(parents, 2)
        child_trucks, child_remaining = crossover_solution(parent1, parent2)
        child_trucks, child_remaining = mutate_solution(child_trucks, child_remaining)
        child_fitness = calculate_fitness(child_trucks, child_remaining)
        next_generation.append((child_fitness, child_trucks, child_remaining))
    return next_generation


def optimize_with_generations(packages):
    population = initialize_population(packages)
    best_solution = max(population, key=lambda x: x[0])
    stagnation_counter = 0

    fitness_history = []

    for generation in range(NUM_GENERATIONS):
        parents = select_parents(population)
        population = create_next_generation(parents)

        current_best = max(population, key=lambda x: x[0])
        if current_best[0] > best_solution[0]:
            best_solution = current_best
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        fitness_history.append(best_solution[0])
        print(f"Generation {generation + 1}: Best Fitness = {best_solution[0]}")

        if stagnation_counter >= NO_IMPROVEMENT_LIMIT:
            print("Stagnation reached. Ending optimization.")
            break

    visualize_statistics(fitness_history, best_solution[1], best_solution[2])
    present_results(best_solution[1], best_solution[2])
    return best_solution


def visualize_statistics(fitness_history, trucks, remaining_packages):
    plt.figure()
    plt.plot(fitness_history, label="Fitness över generationer", color="blue")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitnessförbättring")
    plt.legend()
    plt.grid(True)
    plt.show()

    truck_weights = np.array(
        [sum(p["Vikt"] for p in truck) for truck in trucks.values()]
    )
    truck_profits = np.array(
        [sum(p["Förtjänst"] for p in truck) for truck in trucks.values()]
    )

    x = np.arange(1, NUM_TRUCKS + 1)
    width = 0.35

    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width / 2, truck_weights, width, label="Vikt (kg)", color="blue")
    bars2 = ax.bar(
        x + width / 2, truck_profits, width, label="Förtjänst", color="green"
    )

    ax.set_xlabel("Bil")
    ax.set_ylabel("Värde")
    ax.set_title("Vikt och Förtjänst per Bil")
    ax.set_xticks(x)
    ax.legend()

    plt.tight_layout()
    plt.show()


def present_results(trucks, remaining_packages):
    total_profit = np.sum([p["Förtjänst"] for truck in trucks.values() for p in truck])
    total_weight = np.sum([p["Vikt"] for truck in trucks.values() for p in truck])
    total_delivered = np.sum([len(truck) for truck in trucks.values()])
    remaining_count = len(remaining_packages)
    remaining_profit = np.sum([p["Förtjänst"] for p in remaining_packages])
    total_penalty = np.sum(
        [-(p["Deadline"] ** 2) for p in remaining_packages if p["Deadline"] < 0]
    )

    print("\nRESULT SUMMARY:")
    for truck_id, truck in trucks.items():
        truck_weight = np.sum([p["Vikt"] for p in truck])
        truck_profit = np.sum([p["Förtjänst"] for p in truck])
        print(
            f"Truck {truck_id + 1}: Weight = {truck_weight:.1f} kg, Profit = {truck_profit:.1f}"
        )

    print(f"\nTotal Packages Delivered: {total_delivered}")
    print(f"Total Profit: {total_profit:.1f}")
    print(f"Total Weight: {total_weight:.1f} kg")
    print(f"Remaining Packages: {remaining_count}")
    print(f"Profit from Remaining Packages: {remaining_profit:.1f}")
    print(f"Total Penalty: {total_penalty:.1f}")


if __name__ == "__main__":
    file_path = Path("lagerstatus.csv")
    packages = read_data(file_path)
    optimize_with_generations(packages)
