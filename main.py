import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

MAX_WEIGHT = 800
NUM_TRUCKS = 10
NUM_ITER = 100
WEIGHT_PENALTY = 0.01
DEADLINE_PRIORITY = 1.5
NO_IMPROVEMENT_LIMIT = 200
RESTART_LIMIT = 10
IMPROVEMENT_THRESHOLD = 0.0001


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
        packages, key=lambda x: (x["Deadline"] <= 0, -x["Förtjänst"], x["Deadline"])
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


def calculate_fitness(trucks: dict, remaining_packages: list) -> float:
    total_profit = sum(p["Förtjänst"] for truck in trucks.values() for p in truck)
    total_penalty = sum(
        -(p["Deadline"] ** 2) for p in remaining_packages if p["Deadline"] < 0
    )
    unused_capacity = sum(
        MAX_WEIGHT - sum(p["Vikt"] for p in truck) for truck in trucks.values()
    )
    return total_profit + total_penalty - (WEIGHT_PENALTY * unused_capacity)


def mutate_solution(trucks, remaining_packages, mutation_rate=0.1):
    """
    Applicerar mutation på en lösning genom att flytta paket slumpmässigt.
    """
    all_packages = sum(trucks.values(), []) + remaining_packages
    num_packages = len(all_packages)

    if random.random() < mutation_rate:
        # Flytta ett slumpmässigt paket från en bil till lagret
        truck_id = random.choice(list(trucks.keys()))
        if trucks[truck_id]:
            package = trucks[truck_id].pop(random.randint(0, len(trucks[truck_id]) - 1))
            remaining_packages.append(package)

    if random.random() < mutation_rate:
        # Flytta ett slumpmässigt paket från lagret till en bil
        if remaining_packages:
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

    return trucks, remaining_packages


def optimize_packages_with_restart(packages: list):
    best_fitness = float("-inf")
    best_solution = None
    stagnation_counter = 0
    restart_counter = 0

    for iteration in range(NUM_ITER):
        if best_solution:
            trucks, remaining_packages = best_solution
            all_packages = sum(trucks.values(), []) + remaining_packages
            if iteration < 50:
                random.shuffle(all_packages)
            else:
                all_packages = sorted(
                    all_packages, key=lambda x: (-x["Förtjänst"], x["Deadline"])
                )
        else:
            all_packages = packages

        trucks = allocate_packages(all_packages)
        remaining_packages = [
            p for p in all_packages if p not in sum(trucks.values(), [])
        ]

        # Applicera mutation på lösningen
        trucks, remaining_packages = mutate_solution(
            trucks, remaining_packages, mutation_rate=0.1
        )

        fitness = calculate_fitness(trucks, remaining_packages)

        if fitness > best_fitness:
            best_fitness = fitness
            best_solution = (trucks, remaining_packages)
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        print(f"Iteration {iteration + 1}: Best Fitness = {best_fitness}")

        if stagnation_counter >= NO_IMPROVEMENT_LIMIT:
            print(
                f"Stagnation efter {iteration + 1} generationer. Startar om population..."
            )
            random.shuffle(packages)
            stagnation_counter = 0
            restart_counter += 1

            if restart_counter >= RESTART_LIMIT:
                print("Max antal omstarter uppnått. Avslutar optimering.")
                break

        if (
            iteration > 10
            and abs(fitness - best_fitness) < IMPROVEMENT_THRESHOLD * best_fitness
        ):
            print(
                f"Optimeringen avstannar efter {iteration + 1} generationer (upprepade små förbättringar)."
            )
            break

    visualize_solution(best_solution)
    present_results(best_solution)
    return best_solution


def visualize_solution(solution):
    trucks, _ = solution
    truck_weights = [sum(p["Vikt"] for p in truck) for truck in trucks.values()]
    truck_profits = [sum(p["Förtjänst"] for p in truck) for truck in trucks.values()]

    ind = range(1, NUM_TRUCKS + 1)
    width = 0.35

    plt.bar(ind, truck_weights, width, label="Vikt (kg)", color="blue", alpha=0.7)
    plt.bar(
        [i + width for i in ind],
        truck_profits,
        width,
        label="Förtjänst",
        color="green",
        alpha=0.7,
    )
    plt.title("Vikt och Förtjänst per Budbil")
    plt.xlabel("Budbilar")
    plt.ylabel("Värde")
    plt.legend()
    plt.tight_layout()
    plt.show()


def present_results(solution):
    trucks, remaining_packages = solution
    total_profit = sum(p["Förtjänst"] for truck in trucks.values() for p in truck)
    total_weight = sum(p["Vikt"] for truck in trucks.values() for p in truck)
    total_delivered = sum(len(truck) for truck in trucks.values())
    remaining_count = len(remaining_packages)
    total_penalty = sum(
        -(p["Deadline"] ** 2) for p in remaining_packages if p["Deadline"] < 0
    )

    print("\nRESULTATREDOVISNING:")
    for truck_id, truck in trucks.items():
        truck_weight = sum(p["Vikt"] for p in truck)
        truck_profit = sum(p["Förtjänst"] for p in truck)
        print(
            f"Bil {truck_id + 1}: Vikt = {truck_weight:.1f} kg, Förtjänst = {truck_profit:.1f} kr."
        )

    print(f"\n{remaining_count} st försenade varor kvar.")
    print(f"Total daglig förtjänst: {total_profit:.1f} kr.")
    print(f"Total straffavgift på grund av förseningar: {total_penalty:.1f} kr.")
    print(f"Totalt levererade paket: {total_delivered} paket.")


if __name__ == "__main__":
    file_path = Path("lagerstatus.csv")
    packages = read_data(file_path)
    optimize_packages_with_restart(packages)
