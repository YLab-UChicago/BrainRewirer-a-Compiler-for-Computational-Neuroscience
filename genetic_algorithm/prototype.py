import random
import numpy as np

# Helper function to calculate Euclidean distance (can be replaced with a more appropriate function)
def D(i, j, n):
    return np.sqrt((i // n - j // n) ** 2 + (i % n - j % n) ** 2)

# Objective function to minimize
def ObjectiveFunction(individual, M, n):
    return sum((1 if M[i][j] != 0 else 0) * D(individual[i], individual[j], n) for i in range(n) for j in range(n))

# Function to create a random permutation
def RandomPermutation(n):
    perm = list(range(n))  # Range should be from 0 to n-1
    random.shuffle(perm)
    return perm


def save_matrix_to_file(matrix, filename):
    with open(filename, 'a') as file:  # 'a' mode to append to the file
        for row in matrix:
            file.write(' '.join(map(str, row)) + '\n')
        file.write('\n')  # Add a blank line to separate matrices

def save_message_to_file(message, filename):
    with open(filename, 'a') as file:  # 'a' mode to append to the file
        file.write(message + '\n')


def OrderCrossover(parent1, parent2):
    size = len(parent1)
    proportion = random.uniform(0.3, 0.7)
    window_size = int(size * proportion)
    start = random.randint(0, size - window_size)
    
    # Create two child chromosomes
    child1 = [None] * size
    child2 = [None] * size

    # Copy a consecutive subsequence from the first parent to the first child
    child1[start:start + window_size] = parent1[start:start + window_size]

    # Copy a consecutive subsequence from the second parent to the second child
    child2[start:start + window_size] = parent2[start:start + window_size]

    # Fill in the remaining positions in child1 from parent2, and in child2 from parent1
    fill_remaining(child1, parent2, start, window_size)
    fill_remaining(child2, parent1, start, window_size)

    return child1, child2

def fill_remaining(child, parent, start, window_size):
    size = len(child)
    p_index = 0
    for i in range(size):
        if i < start or i >= start + window_size:
            while parent[p_index] in child:
                p_index += 1
            child[i] = parent[p_index]

# Swap Mutation function
def SwapMutation(individual, subseq_length):
    size = len(individual)
    if subseq_length >= size:
        subseq_length = size // 2
    idx1, idx2 = random.sample(range(size - subseq_length + 1), 2)
    for i in range(subseq_length):
        individual[idx1 + i], individual[idx2 + i] = individual[idx2 + i], individual[idx1 + i]
    return individual

# Tournament Selection function
def TournamentSelection(population, fitnesses, tournament_size):
    selected = []
    for _ in range(len(population)):
        contenders = random.sample(list(zip(population, fitnesses)), tournament_size)
        winner = min(contenders, key=lambda x: x[1])[0]
        selected.append(winner)
    return selected

def transform_individual_to_matrix(individual, original_matrix):
    n = len(individual)  # Length of the individual should match the dimensions of the matrix
    transformed_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            # Get the new row and column indices from the individual
            new_row = individual[i]
            new_col = individual[j]

            # Assign the value from the original matrix to the new position in the transformed matrix
            transformed_matrix[i, j] = original_matrix[new_row, new_col]

    return transformed_matrix



# Function to select the new population
def SelectNewPopulation(population, new_population, fitnesses):
    # Combine old and new populations
    combined = list(zip(population + new_population, fitnesses + [ObjectiveFunction(ind, M, n) for ind in new_population]))
    # Sort by fitness and select the best
    sorted_population = sorted(combined, key=lambda x: x[1])[:len(population)]
    return [ind for ind, _ in sorted_population]

# Function to get the best individual
def GetBestIndividual(population, fitnesses):
    return min(zip(population, fitnesses), key=lambda x: x[1])[0]

def MatrixTransGA(M, n, pop_size, max_generations, tournament_size, mutation_rate, subseq_length, log_filename):

    population = [RandomPermutation(n) for _ in range(pop_size)]
    best_fitness = float('inf')
    best_individual = None
    save_message_to_file("Initial Matrix:", log_filename)
    save_matrix_to_file(M, log_filename)
    
    for generation in range(1, max_generations + 1):
        fitnesses = [ObjectiveFunction(ind, M, n) for ind in population]

        # Track the best individual and fitness
        gen_best_fitness = min(fitnesses)
        if gen_best_fitness < best_fitness:
            best_fitness = gen_best_fitness
            best_individual = population[fitnesses.index(gen_best_fitness)]

        # Print generation information
        if generation % 10 == 0 or generation == 1:
            print(f"Generation {generation}/{max_generations}: Best Fitness = {best_fitness}")

        selected = TournamentSelection(population, fitnesses, tournament_size)
        new_population = []
        for i in range(0, len(selected), 2):
            child1, child2 = OrderCrossover(selected[i], selected[i+1])
            if random.random() < mutation_rate:
                child1 = SwapMutation(child1, subseq_length)
            if random.random() < mutation_rate:
                child2 = SwapMutation(child2, subseq_length)
            new_population.extend([child1, child2])
        
        population = SelectNewPopulation(population, new_population, fitnesses)
        if generation % 1000 == 0:
            # Transform best_individual into matrix form
            current_matrix = transform_individual_to_matrix(best_individual, M)
            save_message_to_file(f"generation {generation}", log_filename)
            save_matrix_to_file(current_matrix, log_filename)

    print("Final Generation Complete")
    print(f"Best Individual: {best_individual}")
    print(f"Best Fitness: {best_fitness}")
    return best_individual

def initialize_sparse_clustered_matrix(size, cluster_size, num_clusters, log_filename):
    matrix = np.zeros((size, size))

    # Place clusters
    for _ in range(num_clusters):
        cluster_value = random.uniform(0.1, 1)  # Assign a random value to the cluster
        x = random.randint(0, size - cluster_size)
        y = random.randint(0, size - cluster_size)
        matrix[x:x+cluster_size, y:y+cluster_size] = cluster_value
    save_message_to_file("Ground Truth:", log_filename)
    save_matrix_to_file(matrix, log_filename)
    # Shuffle the matrix to randomize the location of clusters
    flat_matrix = matrix.flatten()
    np.random.shuffle(flat_matrix)
    shuffled_matrix = flat_matrix.reshape(size, size)

    return shuffled_matrix


# Example usage
n = 16 # Dimension of the neuron matrix
# M = np.random.rand(n, n) # Example matrix M
pop_size = 50
max_generations = 10000
tournament_size = 5
mutation_rate = 0.2
subseq_length = 3


cluster_size = 4  # Size of each cluster
num_clusters = 4  # Number of clusters


log_filename = "log.txt"


M = initialize_sparse_clustered_matrix(n, cluster_size, num_clusters, log_filename)
best_solution = MatrixTransGA(M, n, pop_size, max_generations, tournament_size, mutation_rate, subseq_length, log_filename)