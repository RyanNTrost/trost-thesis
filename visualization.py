from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def plot_output(predicted, expected, epoch, loss, folder):
    n = int(np.sqrt(len(predicted)))
    n_interp = 30
    X1 = np.linspace(0, 0.9, n_interp)
    X2 = np.linspace(0, 0.9, n_interp)
    X1, X2 = np.meshgrid(X1,X2)

    points = np.zeros((n*n, 2))
    for i in range(n):
        for j in range(n):
            points[n*i + j] = [i/n, j/n]

    predicted_interpolation = griddata(points, predicted, (X1, X2), method='cubic')    
    expected_interpolation = griddata(points, expected, (X1, X2), method='cubic')
    
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(X1, X2, predicted_interpolation, cmap=cm.viridis, linewidth=0, antialiased=False)
    ax1.set_title('Predicted Output', y=-0.1)

    ax2 = fig.add_subplot(122, projection='3d', sharex=ax1, sharey=ax1, sharez=ax1)
    surf2 = ax2.plot_surface(X1, X2, expected_interpolation, cmap=cm.viridis, linewidth=0, antialiased=False)
    ax2.set_title('Expected Output', y=-0.1)
    
    fig.suptitle('Epoch: ' + str(epoch) + " Loss: " + str(loss), fontsize=15)
    
    plt.tight_layout()
    plt.savefig(folder + "/epoch-" + str(epoch) + ".png",bbox_inches='tight')
    plt.close(fig)
    plt.show()