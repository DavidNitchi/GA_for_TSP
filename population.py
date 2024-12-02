import random
import pandas as pd
import numpy as np
import pickle
#from .chromosome import Chromosome
class Chromosome():
    #path is a list of ints representing the cities, the order in the path list is te order of visitation
    def __init__(self, path: list[int], fitness: float):
        self.path = path
        self.fitness = fitness
    def mutate(self):
        [c1, c2] = random.sample(self.path, 2)
        self.path[c1], self.path[c2] = self.path[c2], self.path[c1]
    def getPath(self):
        return list(self.path)
    
class Population():
    def __init__(self, csv_dataset_filepath, pop_size: int):
        self.city_coords = pd.read_csv(csv_dataset_filepath)
        #cities = range(0, len(self.city_coords))
        self.pop = []
        self.pop_size = pop_size
        self.total_pop_dist = 0
    # city1, city2 should be integers representing the row of the city in the data

    def initPop(self):
        cities_list = [d for d in range(0, len(self.city_coords))]
        for i in range(0, self.pop_size):
            cities_path = random.sample(cities_list, len(cities_list))
            fitness = self.get_fitness(cities_path)
            print("fitness of", i, ":", fitness)
            self.pop.append(Chromosome(cities_path, fitness))
            self.total_pop_dist += fitness
        return
    
    def dist_cities(self, city1: int, city2:int):
        return np.sqrt(np.sum(np.array(self.city_coords.iloc[city1] - self.city_coords.iloc[city2])**2))
    
    def get_fitness(self, path: list[int]):
        fitness = self.dist_cities(path[-1], path[0])
        for i in range(0, len(path)-1):
            fitness += self.dist_cities(path[i], path[i+1])
        return fitness

    def selection(self, numSelect: int):
        selected = []
        pop_fitness_probs = [c.fitness/self.total_pop_dist for c in self.pop]
        pop_fitness_probs_distribution = np.cumsum(pop_fitness_probs)
        #print(pop_fitness_probs_distribution)
        for i in range(0, numSelect):
            rand_val =np.random.uniform(0,1,1)
            print("Random value:", rand_val)
            bool_select_arr = pop_fitness_probs_distribution > rand_val
            print(bool_select_arr)
            selected_inds = np.where(bool_select_arr == 1)[0][0]
            #print(selected_inds)
            selected.append(selected_inds)
        return selected
    
p: Population = Population("coordinates.csv", 3)
p.initPop()
#print(p.pop[0].path)
#print(p.city_coords.iloc[0])
#print(p.get_fitness(p.pop[0]))
print(p.total_pop_dist)
print(p.selection(5))
#best_tour_path = pickle.load(open("best_tour.pkl", "rb"))
#c_best = Chromosome(best_tour_path, 6190)
#print(p.get_fitness(c_best.path))
