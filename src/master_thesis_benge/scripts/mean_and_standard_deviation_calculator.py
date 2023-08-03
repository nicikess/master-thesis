import math


def calculate_mean(numbers):
    return sum(numbers) / len(numbers)


def calculate_population_standard_deviation(numbers):
    mean = calculate_mean(numbers)
    variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
    population_standard_deviation = math.sqrt(variance)
    return population_standard_deviation


if __name__ == "__main__":
    # Example lists
    lists = [
        [
            92.3499984741211,
            92.3300018310546,
            92.19999694824219,
            91.9800033569336,
            91.93000030517578,
        ],
        [
            83.36000061035156,
            83.13999938964844,
            82.98999786376953,
            82.7699966430664,
            82.27999877929688,
        ],
        [
            85.7699966430664,
            85.12000274658203,
            84.9000015258789,
            84.87999725341797,
            84.7300033569336,
        ],
    ]

    for i, numbers in enumerate(lists):
        population_standard_deviation = calculate_population_standard_deviation(numbers)

        print(f"List {i + 1}:")
        print(
            "$"
            + str(round(calculate_mean(numbers), 2))
            + "\pm"
            + str(round(calculate_population_standard_deviation(numbers), 2))
            + "$"
        )
