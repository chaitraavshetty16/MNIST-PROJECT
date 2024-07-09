from network import init,train
import random
import torch

no_of_generations = 3
no_of_individuals = 3
mutate_factor = 0.01
individuals = []
layers = ['fc1','fc2']

# Code adapted from :https://medium.com/analytics-vidhya/mnist-classifier-using-genetic-cnn-e1e860ecc2e9


def mutate(new_individual):
    print('Mutating')
    with torch.no_grad():
        modules = new_individual._modules
        for module in modules:
            if module in layers:
                weight = modules[module]._parameters['weight']
                n = random.random()
                if (n < mutate_factor):
                    print("mutating with weight parameter")
                    weight *= random.uniform(-0.5,0.5)
        for module in modules:
            if module in layers:
                bias = modules[module]._parameters['bias']
                n = random.random()
                if (n < mutate_factor):
                    print("mutating with bias parameter")
                    bias *= random.uniform(-0.5,0.5)
    return new_individual


def crossover(individuals):
    print("crossovering")
    new_individuals = []
    new_individuals.append(individuals[0])
    new_individuals.append(individuals[1])
    for i in range(2, no_of_individuals):
        if (i < (no_of_individuals - 2)):
            if (i == 2):
                parentA = random.choice(individuals[:3])
                parentB = random.choice(individuals[:3])
            else:
                parentA = random.choice(individuals[:])
                parentB = random.choice(individuals[:])
            for module in layers:
                temp = parentA._modules[module]._parameters['weight']
                print("Below is temp")
                print(temp)
                parentA._modules[module]._parameters['weight'] = parentB._modules[module]._parameters['weight']
                parentB._modules[module]._parameters['weight'] = temp
                new_individual = random.choice([parentA, parentB])
        else:
            new_individual = random.choice(individuals[:])

        new_individuals.append(mutate(new_individual))
    return new_individuals


def evolve(individuals, losses):
    print("Evolving")
    sorted_y_idx_list = sorted(range(len(losses)), key=lambda x: losses[x])
    individuals = [individuals[i] for i in sorted_y_idx_list]
    new_individuals = crossover(individuals)
    return new_individuals


def init_start_training():
    print("Start training of the model")
    individuals = []
    for i in range(no_of_individuals):
        individuals.append(init())
    for generation in range(no_of_generations):
        losses,individuals = train(individuals)
        individuals = evolve(individuals, losses)
    #Returning an evolved model
    return individuals
