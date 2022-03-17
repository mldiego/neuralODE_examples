import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import savemat
from scipy.io import loadmat

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=10000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true', default=True)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--adjoint', action='store_true', default = False)
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

np.random.seed(2021)
torch.random.manual_seed(2021)

data = loadmat('/home/manzand/Documents/MATLAB/NeuralODEs/bouncing_ball/ball_pos.mat')
tvec = data['tvec'][0]
states = data['states'][0]

y_test = states[0] # Use the first trajectory for testing
t_test = tvec[0]

true_y = torch.tensor(y_test)
t = torch.tensor(t_test).flatten()
true_y0 = torch.tensor(true_y[0])


def get_batch():
    s = np.random.randint(1,len(states))
    batch_y0 = torch.tensor(states[s][0])  # (M, D)
    batch_t = torch.tensor(tvec[s][1:]).flatten() # (T)
    batch_y = torch.tensor(states[s][1:])  # (T, M, D)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.viz:
    makedirs('bouncingBallPosAugpng')
    import matplotlib.pyplot as plt


def visualize(true_y, pred_y, odefunc, itr):

    if args.viz:
        
        fig = plt.figure(figsize=(12, 4), facecolor='white')
        ax_traj = plt.gca()

        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x')
        ax_traj.plot(t.cpu().detach().numpy(), true_y.cpu().detach().numpy()[:, 0], 'g-', label = 'true')
        ax_traj.plot(t.cpu().detach().numpy(), pred_y.cpu().detach().numpy()[:, 0], 'b--', label = 'predicted')
        ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
        ax_traj.legend()

        fig.tight_layout()
        plt.savefig('bouncingBallPosAugpng/{:03d}'.format(itr))
        plt.draw()
        plt.pause(0.001)


class ODEFunc(nn.Module):

    def __init__(self, data_dim, hidden_dim):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.Linear(hidden_dim,hidden_dim),
            nn.Linear(hidden_dim, data_dim),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        # return self.net(y)
        return self.net(y)
    
class ODEnet(nn.Module):
    
    def __init__(self,data_aug = 0):
        super(ODEnet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.ode1 = ODEFunc(data_aug+1,64).to(device)
        self.fc2 = nn.Linear(data_aug+1,32)
        self.ode2 = ODEFunc(32,16)
        self.fc3 = nn.Linear(32,1)
        self.int_time1 = torch.tensor([0, 1]).float()
        self.data_aug = data_aug
        
    def forward(self,x,t):
        # self.int_time1 = t
        if self.data_aug > 0:
            x_aug = torch.zeros(x.shape[0],self.data_aug).to(device).flatten()
            out = torch.cat([x,x_aug],0)
        else:
            out = x
        out = odeint(self.ode1, out, t)
        out = self.fc2(out)
        out = self.relu(out)
        out = odeint(self.ode2, out, self.int_time1)
        out = self.fc3(out[1])
        return out
        


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


if __name__ == '__main__':

    ii = 0
    
    aug_dim = 8
    
    model = ODEnet(aug_dim).to(device)
    model.double()
    
    optimizer = optim.RMSprop(model.parameters(), lr=1e-3, weight_decay=1e-3)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    
    loss_meter = RunningAverageMeter(0.97)
    min_loss = 100000
    names = []
    params = []

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch()
        pred_y = model(batch_y0,batch_t).to(device)
        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_y = model(true_y0,t)
                loss = torch.mean(torch.abs(pred_y - true_y))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                visualize(true_y, pred_y, model, ii)
                ii += 1
                if loss < min_loss:
                    print('Updating weights --- Image'+str(ii-1))
                    min_loss = loss
                    names = []
                    params = []
                    params_orig = []
                    for name,param in model.named_parameters():
                        names.append(name)
                        params.append(param.detach().numpy())
                        params_orig.append(param)
                    for name,param in model.named_buffers():
                        names.append(name)
                        params.append(param.detach().numpy())
                    

        end = time.time()
    
    nn1 = dict({'Wb':params,'names':names})
    savemat("odeffnn_bball_pos_aug.mat",nn1)
