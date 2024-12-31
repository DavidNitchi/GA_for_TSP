import random
import pandas as pd
import numpy as np
import pickle
import os.path
import threading
#from .chromosome import Chromosome

class Chromosome():
    #path is a list of ints representing the cities, the order in the path list is te order of visitation
    def __init__(self, path: list[int], fitness: float, ):
        self.path = path
        self.fitness = fitness
    def mutate(self):
        [c1, c2] = random.sample(self.path, 2)
        self.path[c1], self.path[c2] = self.path[c2], self.path[c1]
    def getPath(self):
        return list(self.path)
    
class Population():
    def __init__(self, csv_dataset_filepath, pop_size: int, num_mutations: int):
        self.city_coords = pd.read_csv(csv_dataset_filepath)
        #cities = range(0, len(self.city_coords))
        self.pop: list[Chromosome] = []
        self.pop_size = pop_size
        self.total_pop_dist = 0
        self.max_num_mutations = num_mutations
    # city1, city2 should be integers representing the row of the city in the data

    def initPop(self):
        cities_list = [d for d in range(0, len(self.city_coords))]
        for _ in range(0, self.pop_size):
            cities_path = random.sample(cities_list, len(cities_list))
            fitness = self.get_fitness(cities_path)
            #print("fitness of", i, ":", fitness)
            #print(len(cities_path))
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
    
    def update_total_fitness(self):
        tot = 0
        for c in self.pop:
            tot += c.fitness
        self.total_pop_dist = tot
        return

    def select(self, max:int, selected_list):
        counter = 0
        stop = max/5
        while counter < stop:
            ind1, ind2 = random.sample(range(0, len(self.pop)), 2)
            if self.pop[ind1].fitness < self.pop[ind2].fitness:
                selected_list.append(ind1)
            else:
                selected_list.append(ind2)
            #print("thread", ind, "increased the counter")
            counter+=1
        return

    def tournament_selection(self, num_select: int):
        selected_inds = []
        threads = []
        for i in range(0, 5):
            tmp = threading.Thread(target=self.select, args=( num_select, selected_inds))
            tmp.start()
            threads.append(tmp)
        for t in threads:
            t.join()
        return selected_inds

    # returns indice of selected individuals in self.pop list
    def selection(self, numSelect: int):
        #worst_inds = self.get_worst_individuals(1)[0]
        #print(worst_inds)
        largest_fitness = self.pop[self.get_worst_individuals(1)[0]].fitness
        selected = []
        pop_fitnesses = largest_fitness - np.array([c.fitness for c in self.pop])
        total = sum(pop_fitnesses)
        pop_fitness_probs = [c_fit/total for c_fit in pop_fitnesses]
        pop_fitness_probs_distribution = np.cumsum(pop_fitness_probs)
        for _ in range(0, numSelect):
            rand_val =np.random.uniform(0,1,1)
            bool_select_arr = pop_fitness_probs_distribution > rand_val
            try:
                selected_ind = np.where(bool_select_arr == 1)[0][0]
            except:
                print("++++++++++ START OF ERROR PRINTS +++++++++++++")
                print("largest fitness", largest_fitness)
                print(np.array([c.fitness for c in self.pop]))
                print(pop_fitnesses)
                print(pop_fitness_probs_distribution)
                print("Random value:", rand_val)
                print(bool_select_arr)
                
            #print(selected_inds)
            selected.append(selected_ind)
        return selected
    
    # for crossover, I select a section within the path and swap them, then keep the remaining cities in the same order as they appear in the other parent
    def crossover(self, c1: Chromosome, c2: Chromosome):
        ind1, ind2 = round(random.uniform(0, len(self.city_coords))), round(random.uniform(0, len(self.city_coords)))
        smaller_ind = min(ind1, ind2)
        higher_ind = max(ind1, ind2)
        cross1 = c1.path[smaller_ind:higher_ind]
        #cross2 = c2.path[smaller_ind:higher_ind]
        extra1 = [city for city in c2.path if city not in cross1]
        #extra2 = [city for city in c2.path if city not in cross2]
        child1 = extra1[:smaller_ind] + cross1[:] + extra1[smaller_ind:]
        #child2 = extra2[:smaller_ind] + cross2[:] + extra2[smaller_ind:]
        #print(len(child1))
        #print(len(child2))
        #return Chromosome(child1, self.get_fitness(child1)), Chromosome(child2, self.get_fitness(child2))
        return Chromosome(child1, self.get_fitness(child1))
    #returns indices of the worst individuals in the self.pop list
    def get_worst_individuals(self, num_to_replace: int):
        fitness_list = [c.fitness for c in self.pop]
        worst_inds = np.argsort(fitness_list)
        return worst_inds[-num_to_replace:]

    def make_offspring(self, max, selected_inds, new_pop):
        counter = 0
        stop = int((max+4)/5)
        #print(stop)
        ind1, ind2 = random.sample(selected_inds, 2)
        while counter < stop:
            new_pop.append(self.crossover(self.pop[ind1], self.pop[ind2]))
            counter += 1
        return
    
    def update_pop(self, num_keep):
        num_keep = round(num_keep)
        num_children = len(self.pop)-num_keep
        selected_inds = self.tournament_selection(num_keep)

        #worst_inds = self.get_worst_individuals(num_select)
        #print(selected_inds)
        threads = []
        new_pop = []
        for _ in range(0, 5):
            #print("Size of thread work:")
            tmp = threading.Thread(target=self.make_offspring, args=(num_children, selected_inds, new_pop))
            tmp.start()
            threads.append(tmp)
        for t in threads:
           t.join()
        #print("length of selected indices:", len(selected_inds))
        #print("length of new pop", len(new_pop))
        self.pop = [self.pop[i] for i in selected_inds[:num_keep]] + new_pop[:num_children]
        #print("length of popultion:", len(self.pop))
        #self.update_total_fitness()
        return
    
    def get_best_individuals(self, num_to_replace):
        fitness_list = [c.fitness for c in self.pop]
        worst_inds = np.argsort(fitness_list)
        return worst_inds[:num_to_replace]
    
    def get_best_fitness(self):
        best = self.pop[self.get_best_individuals(1)[0]]
        #print(len(best.path))
        return self.pop[self.get_best_individuals(1)[0]].fitness
    
    #mutate populatino by taking the mutate_prob, multiplying it by the length of the path and then getting a normal distribution aaround that
    def mutate_pop(self, mutate_prob):
        rands = np.random.uniform(0, 1, self.pop_size)
        bools = rands > mutate_prob
        selected_ind = np.where(bools == 1)[0]
        
        for s in selected_ind:
            #num_mutations = random.randint(1, self.max_num_mutations)
            #for _ in range(0, num_mutations):
            self.pop[s].mutate()
        #self.update_total_fitness()
        return
    
def simulate(pop_size, keep_prob):
    best_fit_list = []
    overall_fit_list = []
    #pop_size = 100
    max_gens = 200
    num_to_keep = pop_size*keep_prob # the number multiplied here is the number of individuals we keep
    mutate_prob = 0.9
    max_num_mutations = 1
    p: Population = Population("coordinates.csv", pop_size, max_num_mutations)
    p.initPop()


    for _ in range(0, max_gens):
        #print(i)
        p.update_pop(num_to_keep)
        p.mutate_pop(mutate_prob)
        p.update_total_fitness()

        best_fit_list.append(p.get_best_fitness())
        overall_fit_list.append(p.total_pop_dist)

    results_df = pd.DataFrame(list(zip(best_fit_list, overall_fit_list)), columns=["best_fitness", "overall_fitness"])
    # FILE NAMING CONVENTION: results_df_GA_[max gens]_[pop size]_[number of population to keep between generations]_[mutation probability]_[max number of mutations]
    fname = ["./results/results_df_GA", str(max_gens), str(pop_size), str(num_to_keep), str(mutate_prob), str(max_num_mutations)]
    fname = '_'.join(fname)
    fname = fname+'.pkl'
    if os.path.isfile(fname):
        print("ERROR ALREADY RAN THIS EXPERIMENT, DANGER MAY HAVE OVREWRITTEN A FILE")
        for i in range(2, 100):
            if not os.path.isfile(fname+'_'+'run'+str(i)):
                pickle.dump(results_df, open(fname+'_'+'run'+str(i), 'wb'))
                print("wrote to file: ", fname+'_'+'run'+str(i))
                break
    else:   
        pickle.dump(results_df, open(fname, 'wb'))
        print("wrote to file: ",fname)
    return
