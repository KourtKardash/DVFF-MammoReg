import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


class VisualizationBuilder:

    @staticmethod
    def curved_grid(mammogram, cartesian_x, cartesian_y, origin_point):
        plt.figure(figsize=(10, 6))
        plt.imshow(mammogram, cmap='gray')
        plt.plot(cartesian_x, cartesian_y, color='red', linewidth=0.5)
        plt.plot(cartesian_x.T, cartesian_y.T, color='blue', linewidth=0.5)
        plt.scatter(origin_point[0], origin_point[1], c='cyan', s=20, label='Origin point')
        plt.title('Curvilinear grid')
        plt.legend()
        plt.show()

    @staticmethod
    def display_curvilinear_coordinates(s_found, t_found, intensities):
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(s_found, t_found, c=intensities, cmap='gray', s=1, alpha=0.7)
        plt.colorbar(scatter, label='Intensity')
        plt.xlabel('s coordinate')
        plt.ylabel('t coordinate')
        plt.title('Curvilinear coordinates intensities')
        plt.xlim(0, 1)
        plt.ylim(1, 0)
        plt.show()
