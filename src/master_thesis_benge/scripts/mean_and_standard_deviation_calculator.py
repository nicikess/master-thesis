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
        [85.15, 84.79, 85.12, 85.49, 85.63],
        [93.85, 93.7, 93.91, 93.98, 94.02],
        [88.69, 88.5, 88.66, 88.73, 88.34],
        [95.38, 95.28, 95.35, 95.39, 95.2],
        [91.11, 91.31, 91.06, 91.1, 90.98],
        [96.38, 96.45, 96.34, 96.36, 96.33],
        [92.23, 92.12, 92.38, 91.96, 92.35],
        [96.83, 96.8, 96.89, 96.73, 96.88],
        [88.72, 88.59, 88.65, 88.46, 88.51],
        [95.39, 95.32, 95.34, 95.29, 95.3],
        [91.31, 91.33, 91.39, 91.32, 91.14],
        [96.47, 96.48, 96.49, 96.48, 96.41],
        [92.38, 92.4, 92.42, 92.29, 92.25],
        [96.92, 96.92, 96.91, 96.88, 96.88],
        [92.62, 92.57, 92.66, 92.38, 92.67],
        [97.02, 96.99, 97.04, 96.93, 97.03],
        [],
        [],
        [],
        [],
        [],
        [],
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
