import numpy as np
from method import method_v
import functions_package as fpckg
from metrics import count_metrics
import torch
import matplotlib.pyplot as plt

N = fpckg.N
Edge = fpckg.Edge


# Попробовать взять реализованный градиентный метод, рассчитанный на большую размерность задачи

def criterion(output, target):
    return torch.mean(torch.abs(output - torch.from_numpy(target)) ** 2)


def forward(u_org, pnt_cnt, edge, gammas, ss=0.5):
    u_res = method_v(u_org, pnt_cnt, edge, 1, gammas, ss, 1)
    return u_res


step_size = 10
n_iter = 300

x = np.linspace(-Edge, Edge, N, endpoint=False)
y = np.linspace(-Edge, Edge, N, endpoint=False)
Y, X = np.meshgrid(x, y)

u_orig = fpckg.multifocal([1, 3], [0.8, -1.5], X, Y)


class Params:
    def __init__(self):
        self.gamma = torch.tensor(torch.randn((N, N)), requires_grad=True)

    def __iter__(self):
        return iter([self.gamma])


p = Params()

count_metrics(p.gamma.detach().numpy())


epochs = torch.arange(n_iter)
losses = torch.zeros(n_iter)
optimizer = torch.optim.Adam(p, lr=0.1)

for i in range(n_iter):
    optimizer.zero_grad()
    # making predictions with forward pass
    Y_pred = forward(u_orig, N, Edge, p.gamma)

    # calculating the loss between original and predicted data points
    loss = criterion(Y_pred, u_orig)

    # backward pass for computing the gradients of the loss w.r.t to learnable parameters
    loss.backward()
    optimizer.step()

    # updateing the parameters after each iteration
    #p.gamma.data = p.gamma.data - step_size * p.gamma.grad.data

    # zeroing gradients after each iteration
    p.gamma.grad.data.zero_()

    losses[i] = loss.item()

    # priting the values for understanding
    if i % 20 == 0:
        print('{}. \t{}'.format(i, loss.item()))

count_metrics(p.gamma.detach().numpy())

plt.plot(epochs, losses, label='Loss')
plt.ylabel("MSE")
plt.xlabel("Epoch")
plt.legend()
plt.show()

# fpckg.visualaize_param_matrix(res)

# count_metrics(res)
