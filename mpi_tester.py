from mpi4py.futures import MPIPoolExecutor
import math 

def main():
    with MPIPoolExecutor(max_workers=5) as executor:
        map_iterable = ((2, n) for n in range(0,10))
        for computed_result in executor.starmap(math.pow, map_iterable):
            print(computed_result)

if __name__ == "__main__":
    main()