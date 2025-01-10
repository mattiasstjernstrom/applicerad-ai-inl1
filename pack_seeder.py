import csv
import random
from pathlib import Path

TOTAL_LINES = 1000


def seed_packages(
    n_iter: int = TOTAL_LINES, target_path: Path = Path("lagerstatus.csv")
) -> None:
    assert 0 < n_iter, "n_iter needs to be a positive integer"
    assert n_iter < 9_000_000_000, "n_iter needs to be less than 9 billion"

    entries = []
    id_num = random.randint(1_000_000_000, 9_999_999_999 - n_iter)
    for _ in range(n_iter):
        id_num += 1
        weight = round((random.randint(10, 150) + random.randint(10, 80)) / 20, 1)
        profit = int((random.randint(1, 10) + random.randint(1, 10)) / 2)
        deadline = int((random.randint(-1, 7) + random.randint(-3, 3)) / 2)
        entries.append(
            {
                "Paket_id": id_num,
                "Vikt": weight,
                "Förtjänst": profit,
                "Deadline": deadline,
            }
        )
        if deadline < 0:
            print(f"Försenat paket: {id_num}, Deadline: {deadline}")

    fieldnames = "Paket_id", "Vikt", "Förtjänst", "Deadline"
    with target_path.open("w", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(entries)


if __name__ == "__main__":
    seed_packages()
