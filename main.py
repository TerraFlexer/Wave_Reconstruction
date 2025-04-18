import numpy as np
import matplotlib.pyplot as plt
import functions_package as fpckg
from method import method_v, method_v_slopes

N = fpckg.N
Edge = fpckg.Edge


x = np.linspace(-Edge, Edge, N, endpoint=False)
y = np.linspace(-Edge, Edge, N, endpoint=False)
Y, X = np.meshgrid(x, y)

z = fpckg.multifocal([1, 3], [0.8, -1.5], X, Y)
# z = multifocal_razr([np.sqrt(3), 3], [0.8, -1.5], [0, 1, 3])
# z = fpckg.spiral(3, 1, X, Y)
# z = gauss(X, Y)

dx = 0.5 * Y
dy = -0.5 * X

dola = 7

for i in range(N):
    for j in range(N):
        if i < N // dola or i > (dola - 1) * N // dola or j < N // dola or j > (dola - 1) * N // dola:
            dx[i][j] = 0
            dy[i][j] = 0

# z = generate_random_multifocal_razr()

# z = add_noise(z, 0.01, N)

#z_approx = method_v(z, N, Edge, 0)
#z_approx1 = method_v(z, N, Edge, 1)
z_approx = method_v_slopes(dx, dy, N, Edge, 0)
z_approx1 = method_v_slopes(dx, dy, N, Edge, 1, 0.75, 0.5)

offs = fpckg.offset(z_approx, z, N, 1)

offs1 = fpckg.offset(z_approx1, z, N, 1)


'''fig1, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf2 = ax.plot_surface(X, Y, fpckg.prepare_for_visual(X, Y, z_approx - offs), cmap='plasma')
ax.set_title('Wave')
ax.view_init(45, 60)
#ax.set_zlim([-2, 2])
plt.show()'''


fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
surf1 = ax.plot_surface(X, Y, fpckg.prepare_for_visual(X, Y, z_approx - offs), cmap='plasma')
ax.set_title('stb=0')
# ax.set_zlim([0, np.max(fm)])
ax.view_init(30, -120)
# fig.colorbar(surf1, location='bottom', shrink=0.6, aspect=7)

ax = fig.add_subplot(122, projection='3d')
surf2 = ax.plot_surface(X, Y, fpckg.prepare_for_visual(X, Y, z_approx1 - offs1), cmap='plasma')
ax.set_title('stb=1')
# ax.set_zlim([0, np.max((z_approx - offs))])
ax.view_init(30, -120)
# fig.colorbar(surf2, location='bottom', shrink=0.6, aspect=7)
plt.show()
