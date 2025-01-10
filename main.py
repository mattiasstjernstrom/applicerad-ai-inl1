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


def sort_packages(packages, by_profit=True, by_deadline=False, descending=True):
    if by_profit and by_deadline:
        return sorted(
            packages,
            key=lambda x: (-x["Förtjänst"], x["Deadline"]),
            reverse=not descending,
        )
    elif by_profit:
        return sorted(packages, key=lambda x: -x["Förtjänst"], reverse=not descending)
    elif by_deadline:
        return sorted(packages, key=lambda x: x["Deadline"], reverse=not descending)
    return packages


def calculate_total(truck, key):
    return sum(p[key] for p in truck)


def calculate_all(trucks, key):
    return [calculate_total(truck, key) for truck in trucks.values()]


def allocate_packages(packages: list) -> dict:
    packages = sort_packages(packages, by_profit=True, by_deadline=True)
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
    remaining_packages = sort_packages(remaining_packages, by_profit=True)

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
    truck_profits = calculate_all(trucks, "Förtjänst")
    remaining_deadlines = [
        p["Deadline"] for p in remaining_packages if p["Deadline"] < 0
    ]

    total_profit = np.sum(truck_profits)
    total_penalty = -np.sum(np.square(remaining_deadlines))

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
    num_mutations = max(1, int(len(remaining_packages) * mutation_rate))
    for _ in range(num_mutations):
        if remaining_packages and random.random() < 0.5:
            package = remaining_packages.pop(random.randrange(len(remaining_packages)))
            truck_id = random.choice(range(NUM_TRUCKS))
            if sum(p["Vikt"] for p in trucks[truck_id]) + package["Vikt"] <= MAX_WEIGHT:
                trucks[truck_id].append(package)
        elif trucks:
            truck_id = random.choice(range(NUM_TRUCKS))
            if trucks[truck_id]:
                package = trucks[truck_id].pop(random.randrange(len(trucks[truck_id])))
                remaining_packages.append(package)
    return trucks, remaining_packages


def check_stagnation(stagnation_counter):
    return stagnation_counter >= NO_IMPROVEMENT_LIMIT


def restart_population(packages):
    random.shuffle(packages)
    trucks, remaining_packages = allocate_packages(packages), []
    return trucks, remaining_packages


def optimize_packages_with_restart(packages):
    best_fitness = float("-inf")
    best_solution = None
    stagnation_counter = 0
    restart_counter = 0
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
            trucks, remaining_packages, MUTATION_RATE
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

        if check_stagnation(stagnation_counter):
            print(
                f"Stagnation efter {iteration + 1} generationer. Startar om population..."
            )
            trucks, remaining_packages = restart_population(packages)
            restart_counter += 1

            if restart_counter >= RESTART_LIMIT:
                print("Max antal omstarter uppnått. Avslutar optimering.")
                break

    plot_fitness(fitness_values)
    return best_solution


def plot_fitness(fitness_values):
    plt.plot(range(len(fitness_values)), fitness_values, marker="o")
    plt.xlabel("Iterationer")
    plt.ylabel("Fitnessvärde")
    plt.title("Fitness-utveckling")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def visualize_solution(trucks):
    truck_weights = calculate_all(trucks, "Vikt")
    truck_profits = calculate_all(trucks, "Förtjänst")

    ind = np.arange(len(trucks))
    width = 0.35

    plt.bar(ind, truck_weights, width, label="Vikt", alpha=0.7, color="blue")
    plt.bar(
        ind + width, truck_profits, width, label="Förtjänst", alpha=0.7, color="green"
    )
    plt.xlabel("Lastbilar")
    plt.ylabel("Värden")
    plt.title("Vikt och förtjänst per lastbil")
    plt.legend()
    plt.tight_layout()
    plt.show()


def present_results(trucks, remaining_packages):
    total_packages = sum(len(truck) for truck in trucks.values()) + len(
        remaining_packages
    )
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
    print(f"Totalt antal paket: {total_packages} paket.")
    print(f"Total daglig förtjänst: {total_profit:.0f}")
    print(f"Totalt levererade paket: {total_delivered} paket.")
    print(f"Total straffavgift på grund av förseningar: {total_penalty:.0f}")
    print(f"Total antal kvarvarande paket i lagret: {remaining_count} paket.")
    print(
        f"Totala kvarvarande förtjänsten i lager (exkl. straff): {remaining_total_profit:.0f}"
    )

    visualize_solution(trucks)


if __name__ == "__main__":
    file_path = Path("lagerstatus.csv")
    packages = read_data(file_path)
    try:
        best_solution = optimize_packages_with_restart(packages)
        trucks, remaining_packages = best_solution
        present_results(trucks, remaining_packages)
    except Exception as e:
        print(f"Ett fel uppstod under optimeringen: {e}")
