from multiprocessing import Pool
import torch

def f(x):
    res = 0
    for i in range(1000000):
        res += x * i
    return res
if __name__ == "__main__":
    print(map(f, range(50)))
    print('done')