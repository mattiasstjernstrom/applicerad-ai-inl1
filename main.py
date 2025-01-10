import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

MAX_WEIGHT = 800
NUM_TRUCKS = 10
NUM_ITER = 200
DEADLINE_PRIORITY = 1.5
NO_IMPROVEMENT_LIMIT = 50
RESTART_LIMIT = 20
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
    truck_weights = [0] * NUM_TRUCKS

    for package in packages:
        for truck_id in range(NUM_TRUCKS):
            if truck_weights[truck_id] + package["Vikt"] <= MAX_WEIGHT:
                trucks[truck_id].append(package)
                truck_weights[truck_id] += package["Vikt"]
                break

    return trucks


def fill_trucks(trucks: dict, remaining_packages: list):
    remaining_packages = sorted(remaining_packages, key=lambda x: -x["Förtjänst"])

    for truck_id, truck in trucks.items():
        truck_weight = sum(p["Vikt"] for p in truck)
        available_space = MAX_WEIGHT - truck_weight

        for package in remaining_packages[:]:
            if package["Vikt"] <= available_space:
                trucks[truck_id].append(package)
                available_space -= package["Vikt"]
                remaining_packages.remove(package)

    return trucks, remaining_packages


def calculate_fitness(trucks, remaining_packages):
    truck_profits = np.array(
        [sum(p["Förtjänst"] for p in truck) for truck in trucks.values()]
    )
    remaining_deadlines = np.array(
        [p["Deadline"] for p in remaining_packages if p["Deadline"] < 0]
    )

    total_profit = np.sum(truck_profits)
    total_penalty = -np.sum(remaining_deadlines**2)

    return total_profit + total_penalty


def generate_initial_population(packages, population_size=5):
    population = []
    for _ in range(population_size):
        random.shuffle(packages)
        trucks = allocate_packages(packages)
        remaining_packages = [p for p in packages if p not in sum(trucks.values(), [])]
        trucks, remaining_packages = fill_trucks(trucks, remaining_packages)
        fitness = calculate_fitness(trucks, remaining_packages)
        population.append((fitness, (trucks, remaining_packages)))
    return max(population, key=lambda x: x[0])[1]


def mutate_solution(trucks, remaining_packages, mutation_rate=0.2):
    for _ in range(int(len(remaining_packages) * mutation_rate)):
        if random.random() < 0.5 and remaining_packages:
            package = remaining_packages.pop(
                random.randint(0, len(remaining_packages) - 1)
            )
            for truck_id in range(NUM_TRUCKS):
                if (
                    sum(p["Vikt"] for p in trucks[truck_id]) + package["Vikt"]
                    <= MAX_WEIGHT
                ):
                    trucks[truck_id].append(package)
                    break
        else:
            truck_id = random.choice(list(trucks.keys()))
            if trucks[truck_id]:
                package = trucks[truck_id].pop(
                    random.randint(0, len(trucks[truck_id]) - 1)
                )
                remaining_packages.append(package)

    return trucks, remaining_packages


def optimize_packages_with_restart(packages):
    best_fitness = float("-inf")
    best_solution = None
    stagnation_counter = 0
    restart_counter = 0
    mutation_rate = MUTATION_RATE
    fitness_values = []

    if best_solution is None:
        best_solution = generate_initial_population(packages)

    for iteration in range(NUM_ITER):
        if best_solution:
            trucks, remaining_packages = best_solution
            all_packages = sum(trucks.values(), []) + remaining_packages
            random.shuffle(all_packages)
        else:
            all_packages = packages

        trucks = allocate_packages(all_packages)
        remaining_packages = [
            p for p in all_packages if p not in sum(trucks.values(), [])
        ]

        trucks, remaining_packages = fill_trucks(trucks, remaining_packages)

        trucks, remaining_packages = mutate_solution(
            trucks, remaining_packages, mutation_rate
        )

        trucks, remaining_packages = fill_trucks(trucks, remaining_packages)

        fitness = calculate_fitness(trucks, remaining_packages)

        if fitness > best_fitness:
            best_fitness = fitness
            best_solution = (trucks, remaining_packages)
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        fitness_values.append(best_fitness)

        print(f"Iteration {iteration + 1}: Best Fitness = {best_fitness}")

        if stagnation_counter >= NO_IMPROVEMENT_LIMIT:
            print(
                f"Stagnation efter {iteration + 1} generationer. Startar om population..."
            )
            random.shuffle(packages)
            trucks, remaining_packages = allocate_packages(packages), []
            restart_counter += 1

            if restart_counter >= RESTART_LIMIT:
                print("Max antal omstarter uppnått. Avslutar optimering.")
                break

    plt.plot(range(len(fitness_values)), fitness_values, marker="o", label="Fitness")
    plt.xlabel("Iterationer")
    plt.ylabel("Fitnessvärde")
    plt.title("Fitness-utveckling över iterationer")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    return best_solution


def calculate_statistics(trucks, remaining_packages):
    truck_weights = [sum(p["Vikt"] for p in truck) for truck in trucks.values()]
    truck_profits = [sum(p["Förtjänst"] for p in truck) for truck in trucks.values()]

    truck_stats = {
        "Vikt": {
            "Medelvärde": np.mean(truck_weights),
            "Varians": np.var(truck_weights),
            "Standardavvikelse": np.std(truck_weights),
        },
        "Förtjänst": {
            "Medelvärde": np.mean(truck_profits),
            "Varians": np.var(truck_profits),
            "Standardavvikelse": np.std(truck_profits),
        },
    }

    remaining_weights = [p["Vikt"] for p in remaining_packages]
    remaining_profits = [p["Förtjänst"] for p in remaining_packages]

    remaining_stats = {
        "Vikt": {
            "Medelvärde": np.mean(remaining_weights) if remaining_weights else 0,
            "Varians": np.var(remaining_weights) if remaining_weights else 0,
            "Standardavvikelse": np.std(remaining_weights) if remaining_weights else 0,
        },
        "Förtjänst": {
            "Medelvärde": np.mean(remaining_profits) if remaining_profits else 0,
            "Varians": np.var(remaining_profits) if remaining_profits else 0,
            "Standardavvikelse": np.std(remaining_profits) if remaining_profits else 0,
        },
    }

    return truck_stats, remaining_stats


def visualize_fitness(fitness_values):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(fitness_values)), fitness_values, marker="o", label="Fitness")
    plt.xlabel("Iterationer")
    plt.ylabel("Fitnessvärde")
    plt.title("Fitness-utveckling över iterationer")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def visualize_solution(trucks, remaining_packages):
    truck_weights = [sum(p["Vikt"] for p in truck) for truck in trucks.values()]
    truck_profits = [sum(p["Förtjänst"] for p in truck) for truck in trucks.values()]
    remaining_weights = [p["Vikt"] for p in remaining_packages]
    remaining_profits = [p["Förtjänst"] for p in remaining_packages]

    ind = np.arange(len(trucks))
    width = 0.35

    plt.bar(ind, truck_weights, width, label="Vikt", color="blue", alpha=0.7)
    plt.bar(
        ind + width, truck_profits, width, label="Förtjänst", color="green", alpha=0.7
    )
    plt.xlabel("Lastbilar")
    plt.ylabel("Värde")
    plt.title("Vikt och Förtjänst per lastbil")
    plt.legend()
    plt.tight_layout()
    plt.show()


def present_results(trucks, remaining_packages):
    total_profit = sum(p["Förtjänst"] for truck in trucks.values() for p in truck)
    total_weight = sum(p["Vikt"] for truck in trucks.values() for p in truck)
    total_delivered = sum(len(truck) for truck in trucks.values())
    remaining_count = len(remaining_packages)
    total_penalty = sum(
        -(p["Deadline"] ** 2) for p in remaining_packages if p["Deadline"] < 0
    )
    remaining_total_profit = sum(p["Förtjänst"] for p in remaining_packages)

    print("\nRESULTATREDOVISNING:")
    for truck_id, truck in trucks.items():
        truck_weight = sum(p["Vikt"] for p in truck)
        truck_profit = sum(p["Förtjänst"] for p in truck)
        print(
            f"Bil {truck_id + 1}: Vikt = {truck_weight:.1f}, Förtjänst = {truck_profit}"
        )

    print("\nSTATISTIK:")
    print(f"Total daglig förtjänst: {total_profit:.0f}")
    print(f"Totalt levererade paket: {total_delivered} paket.")
    print(f"Total straffavgift på grund av förseningar: {total_penalty:.0f}")
    print(f"Total antal kvarvarande paket i lagret: {remaining_count} paket.")
    print(
        f"Totala kvarvarande förtjänsten i lager (exkl. straff): {remaining_total_profit:.0f}"
    )


if __name__ == "__main__":
    file_path = Path("lagerstatus.csv")
    packages = read_data(file_path)
    try:
        best_solution = optimize_packages_with_restart(packages)
        trucks, remaining_packages = best_solution
        truck_stats, remaining_stats = calculate_statistics(trucks, remaining_packages)
        present_results(trucks, remaining_packages)
        visualize_solution(trucks, remaining_packages)
    except Exception as e:
        print(f"Ett fel uppstod under optimeringen: {e}")
