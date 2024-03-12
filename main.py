import numpy as np
import matplotlib.pyplot as plt
import functions_package as fpckg
from method import method_v

N = fpckg.N
Edge = fpckg.Edge


x = np.linspace(-Edge, Edge, N, endpoint=False)
y = np.linspace(-Edge, Edge, N, endpoint=False)
Y, X = np.meshgrid(x, y)


# z = multifocal([1, 3, 12], [0.8, -1.5, -11])
z = fpckg.multifocal([1, 3], [0.8, -1.5], X, Y)
# z = multifocal_razr([np.sqrt(3), 3], [0.8, -1.5], [0, 1, 3])
# z = fpckg.spiral(3, 1, X, Y)
# z = gauss(X, Y)

# z = generate_random_multifocal_razr()

# z = add_noise(z, 0.01, N)

z_approx = method_v(z, N, np.pi, 1, gamma=0.5)

offs = fpckg.offset(z_approx, z, N, fpckg.spiral_flag)


fig1, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf2 = ax.plot_surface(X, Y, fpckg.prepare_for_visual(X, Y, z), cmap='plasma')
ax.set_title('Spiral')
ax.view_init(45, 60)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
surf1 = ax.plot_surface(X, Y, fpckg.prepare_for_visual(X, Y, z), cmap='plasma')
ax.set_title('Original function')
# ax.set_zlim([0, np.max(fm)])
ax.view_init(30, -120)
# fig.colorbar(surf1, location='bottom', shrink=0.6, aspect=7)

ax = fig.add_subplot(122, projection='3d')
surf2 = ax.plot_surface(X, Y, fpckg.prepare_for_visual(X, Y, z_approx - offs), cmap='plasma')
ax.set_title('Method approximation')
# ax.set_zlim([0, np.max((z_approx - offs))])
ax.view_init(30, -120)
# fig.colorbar(surf2, location='bottom', shrink=0.6, aspect=7)
plt.show()
