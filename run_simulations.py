from multiprocessing import Process
from population import simulate

if __name__ == "__main__":
    procs = []
    for pop_size in [750, 1300]:
        for cross_prob in [0.1]:
            for i in range(0, 10):
                p = Process(target=simulate, args=(pop_size, cross_prob))
                p.start()
                procs.append(p)

        for proc in procs:
            proc.join()
 