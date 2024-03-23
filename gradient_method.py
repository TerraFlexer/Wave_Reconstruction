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
        self.gamma = torch.tensor(torch.randn((pnt_cnt, pnt_cnt)) * 0.25 + 0.5, requires_grad=True)
        self.ss = torch.FloatTensor(pnt_cnt, pnt_cnt).uniform_(0.25, 0.75).requires_grad_()

    def __iter__(self):
        return iter([self.gamma, self.ss])

    def forward(self, u_orig):
        u_res = method_v(u_orig, self.pnt_cnt, self.edge, 1, self.gamma, self.ss, torch_flag=1)
        return u_res


X, Y = init_net(N, Edge)

p = Params(N, Edge)

fpckg.save_param_value_in_file("gamma", p.gamma.detach().numpy(), "gradient")
fpckg.save_param_value_in_file("ss", p.ss.detach().numpy(), "gradient")

epochs = 300
batch_size = 10

epoch_arr = []
losses = []
losses5 = []

cur_loss = 0
cur_loss5 = 0

optimizer = torch.optim.Adam(p, lr=0.0002)

criterion = torch.nn.MSELoss()

for i in range(epochs):
    for ind, el in enumerate(np.linspace(0.0, 0.2, num=batch_size)):
        u_orig = torch.from_numpy(fpckg.generate_random_multifocal(X, Y))
        u = fpckg.add_noise(u_orig, el, N)

        # zeroing gradients before each iteration
        optimizer.zero_grad()

        # making predictions with forward pass
        pred = p.forward(u)

        # calculating the loss between original and predicted data points
        offs = fpckg.offset(pred, u_orig, N, fpckg.spiral_flag, 1)
        loss = criterion(pred - offs, u_orig)

        # backward pass for computing the gradients of the loss w.r.t to learnable parameters
        loss.backward()

        # updateing the parameters after each iteration
        optimizer.step()

        # zeroing gradients after each iteration
        p.gamma.grad.data.zero_()
        p.ss.grad.data.zero_()

        cur_loss += loss.item()

        pred5 = torch.from_numpy(method_v(u, N, Edge, 1, 0.5))
        offs = fpckg.offset(pred5, u_orig, N, fpckg.spiral_flag, 1)
        loss5 = criterion(pred5 - offs, u_orig)

        cur_loss5 += loss5.item()

    # printing the values for understanding
    if (i + 1) % 10 == 0:
        epoch_arr.append(i)

        cur_loss /= batch_size * 10
        cur_loss5 /= batch_size * 10

        losses.append(cur_loss)
        losses5.append(cur_loss5)

        print('{}. \t{} \t{}'.format(i, cur_loss, cur_loss5))

        cur_loss = 0
        cur_loss5 = 0

plt.plot(epoch_arr, losses, label='Loss gamma+ss')
plt.plot(epoch_arr, losses5, label='Loss 0.5')
plt.ylabel("MSE")
plt.xlabel("Epoch")
plt.legend()
plt.show()

count_metrics(p.gamma.detach().numpy(), p.ss.detach().numpy())

fpckg.visualaize_param_matrix(p.gamma.detach().numpy(), 'gamma')

fpckg.visualaize_param_matrix(p.ss.detach().numpy(), 'ss')

fpckg.save_param_value_in_file("gamma", p.gamma.detach().numpy(), "gradient")
fpckg.save_param_value_in_file("ss", p.ss.detach().numpy(), "gradient")
