import random
import time
from mpi4py import MPI

def monte_carlo(num_points, rank, size):
    """Monte Carlo simulation for estimating Pi"""
    points_per_process = num_points // size
    random.seed(rank + time.time())  # Add time to the seed for true randomness
    inside_circle = 0
    points = []

    for _ in range(points_per_process):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        points.append((x, y))
        if x**2 + y**2 <= 1:
            inside_circle += 1

    return inside_circle, points

def init_mpi():
    """Initialize and return MPI communicator, rank, and size"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    return comm, rank, size

