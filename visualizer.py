import matplotlib.pyplot as plt

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

