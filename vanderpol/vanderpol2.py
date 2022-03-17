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
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='adams')
parser.add_argument('--data_size', type=int, default=500)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=10000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true', default=True)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--adjoint', action='store_true', default=True)
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

np.random.seed(2021)
torch.random.manual_seed(2021)

# data = loadmat('/home/manzand/Documents/MATLAB/neuralODE/vanderpol/vdp_data_trace.mat')
data = loadmat('/home/manzand/Documents/MATLAB/neuralODE/vanderpol/vdp_data.mat')

tvec = data['tvec'][0]
states = data['states'][0]

# y_test = states[0:10] # Use the first 10 trajectory for testing
# t_test = tvec[0:10]

# true_y = torch.tensor(y_test)
# t = torch.tensor(t_test).flatten()
# # t = t[:500] # Reduce training data to first 2500 points
# # true_y = true_y[:500,:]
# true_y0 = torch.tensor(true_y[0])


# # def get_batch():
# #     s = np.random.randint(1,len(states))
# #     iidx = np.random.randint(1,(len(states[s])-100))
# #     batch_y0 = torch.tensor(states[s][iidx])  # (M, D)
# #     batch_t = torch.tensor(tvec[s][iidx:iidx+100]).flatten() # (T)
# #     batch_y = torch.tensor(states[s][iidx:iidx+100])  # (T, M, D)
# #     return batch_y0.to(device), batch_t.to(device), batch_y.to(device)

# def get_batch(s):
#     s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
#     batch_y0 = true_y[s]  # (M, D)
#     batch_t = t[:args.batch_time]  # (T)
#     batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
#     return batch_y0.to(device), batch_t.to(device), batch_y.to(device)
    
t  = []
true_y = []
true_y0 = []
val_exps = 5
for ttt in range(val_exps):
    true_y0.append(torch.tensor(states[ttt][0]).to(device)) 
    t.append(torch.tensor(tvec[ttt][:]).flatten().to(device))
    true_y.append(torch.tensor(states[ttt][:]).to(device))


def get_batch():
    s = np.random.randint(11,len(states),size = (1,3), dtype = int)
    s = list(s[0][:])
    batch_y0 = []
    batch_t = []
    batch_y = []
    for ii in s:
        batch_y0.append(torch.tensor(states[ii][0]).to(device)) 
        batch_t.append(torch.tensor(tvec[ii][:]).flatten().to(device))
        batch_y.append(torch.tensor(states[ii][:]).to(device))
    #     batch_y0.append(states[ii][0]) 
    #     batch_t.append(tvec[ii][:minL])
    #     batch_y.append(states[ii][:minL])
    # batch_y0 = torch.tensor(batch_y0)
    # batch_y = torch.tensor(batch_y)
    # batch_t = torch.tensor(batch_t)
    return batch_y0, batch_t, batch_y


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.viz:
    makedirs('VDP2png')
    import matplotlib.pyplot as plt



def visualize(true_y, pred_y, odefunc, itr):

    if args.viz:
        
        fig = plt.figure(figsize=(12, 4), facecolor='white')
        ax_traj = fig.add_subplot(121)
        ax_phase = fig.add_subplot(122)
        # plt.show()

        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        ax_traj.plot(t.cpu().detach().numpy(), true_y.cpu().detach().numpy()[:, 0], t.cpu().numpy(), true_y.cpu().detach().numpy()[:, 1], 'g-', label = 'true')
        ax_traj.plot(t.cpu().detach().numpy(), pred_y.cpu().detach().numpy()[:, 0], '--', t.cpu().numpy(), pred_y.cpu().detach().numpy()[:, 1], 'b--', label = 'predicted')
        ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
        ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('y')
        ax_phase.plot(true_y.cpu().detach().numpy()[:, 0], true_y.detach().numpy()[:, 1], 'g-')
        ax_phase.plot(pred_y.cpu().detach().numpy()[:, 0], pred_y.detach().numpy()[:, 1], 'b--')

        fig.tight_layout()
        plt.savefig('VDP2png/{:03d}'.format(itr))
        plt.draw()
        plt.pause(0.001)


class ODEFunc(nn.Module):

    def __init__(self, data_dim, hidden_dim):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.LeakyReLU(), # layer 1
            nn.Linear(hidden_dim,hidden_dim),
            nn.LeakyReLU(), # layer 2
            nn.Linear(hidden_dim,hidden_dim),
            nn.LeakyReLU(), # layer 3
            nn.Linear(hidden_dim,hidden_dim),
            nn.LeakyReLU(), #l layer 4
            nn.Linear(hidden_dim, data_dim), #layer 5
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        # return self.net(y)
        return self.net(y)
    
class ODEFunc2(nn.Module):

    def __init__(self, data_dim, hidden_dim):
        super(ODEFunc2, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim,data_dim),
            # nn.Tanh(),
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
        # self.relu = nn.ReLU(inplace=True)
        # self.odeL = ODEFunc2(2,6).to(device)
        # self.fc1 = nn.Linear(2,6)
        self.ode1 = ODEFunc(2,50).to(device)
        # self.fc2 = nn.Linear(6,2)
        # self.ode2 = ODEFunc(32,16)
        # self.fc3 = nn.Linear(32,1)
        # self.int_time1 = torch.tensor([0, 1]).float()
        self.data_aug = data_aug
        self.int_timeL = torch.tensor([0, 1]).float()
        
    def forward(self,x,t):
        # self.int_time1 = t
        if self.data_aug > 0:
            if len(x.size()) > 1:
                x_aug = torch.zeros(x.shape[0],self.data_aug).to(device)
                out = torch.cat([x,x_aug],1)
            else:
                x_aug = torch.zeros(1,self.data_aug).to(device).flatten()
                out = torch.cat([x,x_aug],0)   
            # out = torch.transpose(out,0,1)
        else:
            out = x
        # out = odeint(self.odeL,out,self.int_timeL)
        # out = self.fc1(out)
        out = odeint(self.ode1, out, t)
        # out = self.fc2(out)
        # out = self.relu(out)
        # out = odeint(self.ode2, out, self.int_time1)
        # out = self.fc3(out[1])
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

    aug_dim = 0
    
    model = ODEnet(aug_dim).to(device)
    model.double()
    print(model)
    
    optimizer = optim.RMSprop(model.parameters(), lr=1e-3, weight_decay = 1e-10)
    # optimizer = optim.RMSprop(model.parameters(), lr=1e-4)
    # optimizer = optim.Adam(model.parameters(), lr=1e-5)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    
    loss_meter = RunningAverageMeter(0.97)
    min_loss = 100000
    names = []
    params = []

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_y0g, batch_tg, batch_yg = get_batch()
        loss_temp = 0
        for exp in range(1,len(batch_y0g)):
            batch_y0 = batch_y0g[exp]
            batch_t = batch_tg[exp]
            batch_y = batch_yg[exp]
            pred_y = model(batch_y0,batch_t).to(device)
            loss_temp += torch.mean(torch.abs(pred_y - batch_y))/len(batch_t)
        # loss = torch.mean(torch.abs(pred_y - batch_y))
        loss = loss_temp/len(batch_y0g)
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                for ttt in range(val_exps):
                    loss_temp = 0
                    pred_y = model(true_y0[ttt],t[ttt])
                    # visualize(true_y[ttt], pred_y, t[ttt], ii, ttt)
                    loss_temp += torch.mean(torch.abs(pred_y - true_y[ttt]))
                loss = loss_temp/val_exps
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                # visualize(true_y[ttt], pred_y, t[ttt], ii)
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
    savemat("odeffnn_vdp.mat",nn1)
