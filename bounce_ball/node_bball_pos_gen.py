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
parser.add_argument('--niters', type=int, default=500)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true', default=True)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

np.random.seed(2021)
torch.random.manual_seed(2021)

data = loadmat('/home/manzand/Documents/MATLAB/NeuralODEs/bouncing_ball/ball_pos_gen.mat')
tvec = data['tvec'][0]
states = data['states'][0]

# Get minimum length of points in state (or tvec)
# minL = 1000000000000000000000
# for lll in tvec:
#     minL = min(minL,len(lll))

# y_test = states[0] # Use the first 100 trajectories for testing (10%)
# t_test = tvec[0]

# true_y = torch.tensor(y_test)
# t = torch.tensor(t_test).flatten()
# true_y0 = torch.tensor(true_y[0])
t  = []
true_y = []
true_y0 = []
val_exps = 50
for ttt in range(val_exps):
    true_y0.append(torch.tensor(states[ttt][0]).to(device)) 
    t.append(torch.tensor(tvec[ttt][:]).flatten().to(device))
    true_y.append(torch.tensor(states[ttt][:]).to(device))


def get_batch():
    s = np.random.randint(101,len(states),size = (1,100), dtype = int)
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
    makedirs('bouncingBallPosGenpng')
    import matplotlib.pyplot as plt


def visualize(yt, yp, ts, itr, ett):

    if args.viz:
        
        fig = plt.figure(figsize=(12, 4), facecolor='white')
        ax_traj = plt.gca()

        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x')
        ax_traj.plot(ts.cpu().detach().numpy(), yt.cpu().detach().numpy()[:, 0], label = 'true')
        ax_traj.plot(ts.cpu().detach().numpy(), yp.cpu().detach().numpy()[:, 0], 'b--', label = 'predicted')
        ax_traj.set_xlim(ts.cpu().min(), ts.cpu().max())
        ax_traj.legend()

        fig.tight_layout()
        plt.savefig('bouncingBallPosGenpng/{:03d}_{:03d}'.format(itr,ett))
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
    
    def __init__(self):
        super(ODEnet, self).__init__()
        self.fc1 = nn.Linear(1,64)
        self.relu = nn.ReLU(inplace=True)
        self.ode1 = ODEFunc(64,64).to(device)
        self.fc2 = nn.Linear(64,64)
        self.ode2 = ODEFunc(64,64)
        self.fc3 = nn.Linear(64,1)
        # self.ode3 = ODEFunc(32,32)
        # self.fc4 = nn.Linear(32,1)
        self.int_time1 = torch.tensor([0, 1]).float()
        
    def forward(self,x,t):
        # self.int_time1 = t
        out = x
        out = self.fc1(out)
        out = self.relu(out)
        out = odeint(self.ode1, out, t)
        out = self.fc2(out)
        out = self.relu(out)
        out = odeint(self.ode2, out, self.int_time1)
        out = self.fc3(out[1])
        # out = self.relu(out)
        # out = odeint(self.ode3,out,self.int_time1)
        # out = self.fc4(out[1])
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
        
def test_final_model(odenet,wb_prms,y0_test,y_test,t_test, viz_iter):
    k = 0
    for name,mods in odenet.named_modules():
        if 'linear' in str(type(mods)):
            # mods.weight = torch.nn.Parameter(torch.tensor(params[k]))
            mods.weight = wb_prms[k]
            k+=1
            mods.bias = wb_prms[k]
            k+=1
    pred_y = odenet(y0_test[0],t_test[0])
    loss_test = torch.mean(torch.abs(pred_y - y_test[0]))
    visualize(true_y[ttt], pred_y, t[ttt], viz_iter, viz_iter)
    print('Final loss = ' + str(loss_test))
    return
        
        
if __name__ == '__main__':

    ii = 0

    model = ODEnet().to(device)
    model.double()
    
    optimizer = optim.RMSprop(model.parameters(), lr=1e-4, weight_decay = 1e-3)
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
                    visualize(true_y[ttt], pred_y, t[ttt], ii, ttt)
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
    savemat("odeffnn_bball_pos_gen.mat",nn1)
    
    final_model = ODEnet().to(device).double()
    test_final_model(final_model,params_orig,true_y0,true_y,t,ii+100)
    test_final_model(model,params_orig,true_y0,true_y,t,ii+200)
    
    params2 = []
    params3 = []
    for name,param in model.named_parameters():
        params2.append(param)
    for name,param in final_model.named_parameters():
        params3.append(param.detach().numpy())
    params_diff = []
    for ppp in range(len(params2)):
        params_diff.append(torch.abs(params2[ppp]-torch.tensor(params3[ppp]).to(device)))
    

