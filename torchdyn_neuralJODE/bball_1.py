import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import savemat
from scipy.io import loadmat

from torchdyn.nn.utils import DataControl
from torchdyn.models import NeuralDE
import matplotlib.pyplot as plt
import torch.utils.data as data
import pytorch_lightning as pl


device = torch.device('cpu')

np.random.seed(2021)
torch.random.manual_seed(2021)

data = loadmat('/home/manzand/Documents/MATLAB/NeuralODEs/bouncing_ball/ball_data.mat')
tvec = data['tvec'][0]
states = data['states'][0]

y_test = states[0] # Use the first trajectory for testing
t_test = tvec[0]

true_y = torch.tensor(y_test)
t = torch.tensor(t_test).flatten()
true_y0 = torch.tensor(true_y[0])


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

viz = True
if viz:
    makedirs('bouncingBallpng')
    import matplotlib.pyplot as plt


def visualize(true_y, pred_y, odefunc, itr):

    if viz:
        
        fig = plt.figure(figsize=(12, 4), facecolor='white')
        ax_traj = fig.add_subplot(121)
        ax_phase = fig.add_subplot(122)

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
        plt.savefig('bouncingBallpng/{:03d}'.format(itr))
        plt.draw()
        plt.pause(0.001)


# Define Learner
class Learner(pl.LightningModule):
    def __init__(self, model:nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.005)

    def train_dataloader(self):
        return trainloader

# We choose to divide the domain [0,1] in num_pieces=5 intervals
num_pieces = 5

# stacked depth-invariant Neural ODEs
nde = []
for i in range(num_pieces):
    # nde.append(NeuralDE(nn.Sequential(DataControl(),
    #                                   nn.Linear(4, 4),
    #                                    # nn.Tanh(),
    #                                   nn.Linear(4, 2)), solver='dopri5'))
    nde.append(NeuralDE(nn.Sequential(nn.Linear(2, 4),
                                       # nn.Tanh(),
                                      nn.Linear(4, 2)), solver='dopri5'))
    # In this case the state "jump" is parametrized by a simple linear layer
    nde.append(
        nn.Linear(2, 2)
    )

model = nn.Sequential(*nde).to(device)

learn = Learner(model)
trainer = pl.Trainer(min_epochs=200, max_epochs=250)
trainer.fit(learn)

ii = 0

print(model)

optimizer = optim.RMSprop(model.parameters(), lr=1e-4)
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
savemat("odeffnn_bball.mat",nn1)
