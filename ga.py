import os
from concurrent.futures import ProcessPoolExecutor
from math import log10

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from util import *

# Representation is 3 unsigned ints of length SUBSTRING_PRECISION
# Conv layer 1 kernel count = int(x) + 1
# Conv layer 2 kernel count = int(x) + 1
# FC layer size = linearly map x + 1 to [1, 256] (256 constant max bound, lower bound decreases with increasing precision)


# GA Hyperparameters
MU = 10                  # Parent population size
LAMBDA = 20              # Child population size
SUBSTRING_PRECISION = 4  # Number of bits per substring
MAX_GENERATIONS = 5      # Maximum number of generations to run
CROSSOVER_RATE = 0.7     # Probability of crossover
MUTATION_RATE = 0.2      # Probability of mutation
ALPHA = 0.5              # Weighting of train accuracy vs capacity (higher = more weight on train accuracy)
TARGET_FITNESS = 95      # Max possible fitness, regardless of weighting, is 100

# CNN Hyperparameters
BETAS = (0.9, 0.999)  # For Adam optimizer
LEARN_RATE = 0.001
NUM_CLASSES = 10
NUM_EPOCHS = 10
TRAIN_CONCURRENT = 6  # Number of models to train concurrently
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset
DATA_DIR = 'C:/Users/ryanj/MyFiles/Data/pytorch_datasets'
DATASET = 'mnist'  # 'mnist' or 'cifar'
BATCH_SIZE = 128


def scale(x, old_min, old_max, new_min, new_max):
    return (x - old_min) * (new_max - new_min) / (old_max - old_min) + new_min


def create_random_bitstring(length):
    # Creates a random bitstring of the specified length, return type is str
    bitstring_arr = np.random.randint(low=0, high=2, size=length, dtype=np.uint8)
    return ''.join([str(bit) for bit in bitstring_arr])


def interpret_bitstring(bitstring):
    x = [int(bitstring[i:i+SUBSTRING_PRECISION], 2) for i in range(0, len(bitstring), SUBSTRING_PRECISION)]
    conv1_kernel_count = x[0] + 1
    conv2_kernel_count = x[1] + 1
    fc_layer_size = int(scale(x[2] + 1, 0, 2**SUBSTRING_PRECISION, 1, 256))
    return (conv1_kernel_count, conv2_kernel_count, fc_layer_size)


def fitness(individual):
    train_loader, test_loader = get_loaders(data_dir=DATA_DIR, dataset=DATASET, batch_size=BATCH_SIZE, download=False)

    conv1_count, conv2_count, fc_size = interpret_bitstring(individual)
    model = SmallCNN(num_classes=NUM_CLASSES, conv1_count=conv1_count, conv2_count=conv2_count, fc_size=fc_size)

    optimizer = torch.optim.Adam(model.parameters(), betas=BETAS, lr=LEARN_RATE)

    log_dict = train_model(model=model, num_epochs=NUM_EPOCHS, optimizer=optimizer, device=DEVICE, train_loader=train_loader)

    accuracy = log_dict['train_acc_per_epoch'][-1]
    num_params = model.get_num_params()
    # scaled_num_params = -2.718**(EPSILON * num_params) + 100
    scaled_num_params = -75 * log10(0.0001 * num_params) + 100

    fitness = max(0, ALPHA * accuracy + (1 - ALPHA) * scaled_num_params)
    return {'fitness': fitness, 'accuracy': accuracy, 'num_params': num_params}


def crossover(parent_1, parent_2):
    crossover_point = np.random.randint(low=1, high=len(parent_1) - 1)
    child_1 = parent_1[:crossover_point] + parent_2[crossover_point:]
    child_2 = parent_2[:crossover_point] + parent_1[crossover_point:]
    return child_1, child_2


def mutate(bitstring, mutation_rate):
    bitstring = [int(bit) for bit in bitstring]
    for i in range(len(bitstring)):
        if np.random.random() < mutation_rate:
            bitstring[i] = int(not (bitstring[i]))
    return ''.join([str(bit) for bit in bitstring])


def run_ga():
    # Make sure data is downloaded before running
    get_loaders(data_dir=DATA_DIR, dataset=DATASET)

    population = [create_random_bitstring(3 * SUBSTRING_PRECISION) for _ in range(MU)]
    df = pd.DataFrame(columns=['Generation', 'Champion Bitstring', 'Champion Values',
                               'Champion Fitness', 'Average Fitness', 'Minimum Fitness',
                               'Champion Accuracy', 'Average Accuracy', 'Minimum Accuracy',
                               'Champion Capacity', 'Average Capacity', 'Maximum Capacity'])

    for generation in range(1, MAX_GENERATIONS + 1):
        print(f'Generation {generation}:')

        # Crossover
        children_population = []
        while len(children_population) < LAMBDA:
            if np.random.random() < CROSSOVER_RATE:
                parent_1, parent_2 = np.random.choice(population, size=2, replace=False)
                child_1, child_2 = crossover(parent_1, parent_2)
                children_population.extend((child_1, child_2))
        population.extend(children_population)

        # Mutation
        for i in range(len(population)):
            population[i] = mutate(population[i], MUTATION_RATE)

        # Survivor selection
        with ProcessPoolExecutor(TRAIN_CONCURRENT) as executor:
            results = list(tqdm(executor.map(fitness, population), total=len(population)))
            individual_stats = list(results)

        pop_with_stats=[
            (
                population[i],
                individual_stats[i]['fitness'],
                individual_stats[i]['accuracy'],
                individual_stats[i]['num_params'],
            )
            for i in range(len(population))
        ]
        pop_with_stats.sort(key=lambda x: x[1], reverse=True)
        population = [pop_with_stats[i][0] for i in range(MU)]

        # Statistics
        champion = {'bitstring': pop_with_stats[0][0],
                    'values': interpret_bitstring(pop_with_stats[0][0]),
                    'fitness': pop_with_stats[0][1],
                    'accuracy': pop_with_stats[0][2],
                    'num_params': pop_with_stats[0][3]}

        avg_capacity = np.mean([individual['num_params'] for individual in individual_stats])
        avg_accuracy = np.mean([individual['accuracy'] for individual in individual_stats])
        avg_fitness = np.mean([individual['fitness'] for individual in individual_stats])

        max_capacity = np.max([individual['num_params'] for individual in individual_stats])
        min_accuracy = np.min([individual['accuracy'] for individual in individual_stats])
        min_fitness = np.min([individual['fitness'] for individual in individual_stats])

        print(f'Champ {champion["values"]} fit {champion["fitness"]} acc {champion["accuracy"]} cap {champion["num_params"]} | Avg fit {avg_fitness} | Avg cap {avg_capacity} | Avg acc {avg_accuracy}', end='\n')

        df.loc[generation] = [generation, champion['bitstring'], champion['values'],
                              champion['fitness'], avg_fitness, min_fitness,
                              champion['accuracy'], avg_accuracy, min_accuracy,
                              champion['num_params'], avg_capacity, max_capacity]

        # Termination condition
        if avg_fitness >= TARGET_FITNESS:
            return df
    return df


if __name__ == '__main__':
    results_df = run_ga()
    print(results_df)
    results_df.to_csv(os.path.join('results', f'ga_{DATASET}_{NUM_EPOCHS}_epochs_{MAX_GENERATIONS}_generations.csv'), index=False)

