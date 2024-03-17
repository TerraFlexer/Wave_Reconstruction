import numpy as np
from method import method_v, init_net
import functions_package as fpckg
from metrics import count_metrics
import torch
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error as mse

N = fpckg.N
Edge = fpckg.Edge


class Params:
    def __init__(self, pnt_cnt, edge):
        self.pnt_cnt = pnt_cnt
        self.edge = edge
        self.gamma = torch.tensor(torch.randn((pnt_cnt, pnt_cnt)) * 0.5 + 0.25, requires_grad=True)
        # self.ss = torch.tensor(torch.randn((pnt_cnt, pnt_cnt)), requires_grad=True)

    def __iter__(self):
        return iter([self.gamma])

    def forward(self, u_orig):
        u_res = method_v(u_orig, self.pnt_cnt, self.edge, 1, self.gamma, torch_flag=1)
        return u_res


'''def criterion(output, target):
    return torch.nn.MSELoss(output, target)'''


X, Y = init_net(N, Edge)

u_org = torch.from_numpy(fpckg.add_noise(fpckg.multifocal([1, 3], [0.8, -1.5], X, Y), 0.1, N))

p = Params(N, Edge)

# count_metrics(p.gamma.detach().numpy())

epochs = 400
batch_size = 5

epoch_arr = []
losses = []
losses5 = []

cur_loss = 0
cur_loss5 = 0

optimizer = torch.optim.Adam(p, lr=0.00005)

criterion = torch.nn.MSELoss()

for i in range(epochs):
    for ind, el in enumerate(np.linspace(0.0, 0.2, num=batch_size)):
        u_org = torch.from_numpy(fpckg.add_noise(fpckg.multifocal([1, 3], [0.8, -1.5], X, Y), el, N))

        # zeroing gradients before each iteration
        optimizer.zero_grad()

        # making predictions with forward pass
        pred = p.forward(u_org)

        # calculating the loss between original and predicted data points
        offs = fpckg.offset(pred, u_org, N, fpckg.spiral_flag, 1)
        loss = criterion(pred - offs, u_org)

        # backward pass for computing the gradients of the loss w.r.t to learnable parameters
        loss.backward()

        # updateing the parameters after each iteration
        optimizer.step()

        # zeroing gradients after each iteration
        p.gamma.grad.data.zero_()

        cur_loss += loss.item()

        pred5 = torch.from_numpy(method_v(u_org, N, Edge, 1, 0.5))
        offs = fpckg.offset(pred5, u_org, N, fpckg.spiral_flag, 1)
        loss5 = criterion(pred5 - offs, u_org)

        cur_loss5 += loss5.item()

    # printing the values for understanding
    if i % 20 == 0:
        epoch_arr.append(i)
        losses.append(cur_loss / batch_size / 20)
        losses5.append(cur_loss5 / batch_size / 20)

        cur_loss = 0
        cur_loss5 = 0

        pred5 = torch.from_numpy(method_v(u_org, N, Edge, 1, 0.5))
        loss5 = criterion(pred5, u_org)
        print('{}. \t{} \t{}'.format(i, losses[i], losses5[i]))

plt.plot(epoch_arr, losses, label='Loss gamma')
plt.plot(epoch_arr, losses5, label='Loss 0.5')
plt.ylabel("MSE")
plt.xlabel("Epoch")
plt.legend()
plt.show()

count_metrics(p.gamma.detach().numpy())

# fpckg.visualaize_param_matrix(res)

# count_metrics(res)
