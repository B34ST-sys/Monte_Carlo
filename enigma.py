import random
import time
import sys
import matplotlib.pyplot as plt
from mpi4py import MPI

# Heartbeat timeout configuration
HEARTBEAT_TIMEOUT = 10  # Seconds
HEARTBEAT_INTERVAL = 2  # Seconds

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

def update_visualization(all_points):
    """Update the visualizer with the latest points."""
    if plt.fignum_exists(1):
        plt.clf()
        inside_x = [x for x, y in all_points if x**2 + y**2 <= 1]
        inside_y = [y for x, y in all_points if x**2 + y**2 <= 1]
        outside_x = [x for x, y in all_points if x**2 + y**2 > 1]
        outside_y = [y for x, y in all_points if x**2 + y**2 > 1]

        circle = plt.Circle((0, 0), 1, color='b', fill=False, linewidth=2)
        plt.gca().add_artist(circle)
        plt.scatter(inside_x, inside_y, color='green', s=1, label='Inside Circle')
        plt.scatter(outside_x, outside_y, color='red', s=1, label='Outside Circle')

        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend()
        plt.draw()
        plt.pause(0.001)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print("[Root] Program Started.")
        while True:
            print("Select Mode:")
            print("1. Run for a specific number of points")
            print("2. Run endlessly until a node fails")
            try:
                choice = int(input("Enter choice (1 or 2): ").strip())
                if choice in [1, 2]:
                    break
                else:
                    print("[Root] Invalid choice. Please enter 1 or 2.")
            except ValueError:
                print("[Root] Invalid input. Please enter a numeric value (1 or 2).")

        if choice == 1:
            num_points = int(input("Enter the number of points for the Monte Carlo simulation: ").strip())
        elif choice == 2:
            num_points = int(input("Enter the number of points per iteration for each process: ").strip())
    else:
        choice = None
        num_points = None

    choice = comm.bcast(choice, root=0)
    num_points = comm.bcast(num_points, root=0)
    iteration_count = 0
    running = True

    if rank == 0:
        plt.ion()
        fig = plt.figure(figsize=(6, 6))

    try:
        if choice == 2:
            if rank == 0:
                print("[Root] Starting Monte Carlo simulation indefinitely.")
            
            while running:
                if rank == 0 and not plt.fignum_exists(fig.number):
                    running = False
                
                running = comm.bcast(running, root=0)
                if not running:
                    break
                
                start_time = time.time()
                inside_circle, points = monte_carlo(num_points, rank, size)
                
                all_points = comm.gather(points, root=0)
                
                total_inside = comm.reduce(inside_circle, op=MPI.SUM, root=0)
                if total_inside is None:
                    total_inside = 0
                
                pi_estimate = 4 * total_inside / num_points
                iteration_count += 1
                elapsed_time = time.time() - start_time
                
                if rank == 0:
                    all_points = [p for sublist in all_points for p in sublist]  # Flatten list
                    update_visualization(all_points)
                    sys.stdout.write(f"\r[Root] Iteration {iteration_count} - Estimated Pi: {pi_estimate:.9f} "
                                     f"Points inside the circle: {total_inside} "
                                     f"Time: {elapsed_time:.6f} seconds   ")
                    sys.stdout.flush()

                time.sleep(0.5)

        elif choice == 1:
            start_time = time.time()
            inside_circle, points = monte_carlo(num_points, rank, size)
            all_points = comm.gather(points, root=0)
            total_inside = comm.reduce(inside_circle, op=MPI.SUM, root=0)
            if rank == 0:
                all_points = [p for sublist in all_points for p in sublist]  # Flatten list
                update_visualization(all_points)
                pi_estimate = 4 * total_inside / num_points
                elapsed_time = time.time() - start_time
                print(f"\n[Root] Estimated Pi = {round(pi_estimate, 9)} in {elapsed_time:.6f} seconds")
                update_visualization(all_points)
                print("[Root] Close the visualization window to exit.")
                plt.ioff()
                plt.show()
    
    except KeyboardInterrupt:
        print("\n[Root] Simulation interrupted.")
    finally:
        running = False
        comm.bcast(running, root=0)
        comm.Barrier()
        if rank == 0:
            print("[Root] Finished execution.")
            plt.ioff()
            plt.close()
            sys.exit(0)

if __name__ == "__main__":
    main()

