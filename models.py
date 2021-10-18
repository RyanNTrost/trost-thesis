from torch import nn, optim, save
import time
import numpy as np
from visualization import plot_output
from laplace_dataset import idct2

class LaplaceModel(nn.Module):
    def __init__(self, n, input_size, output_size, isDCT, device, learning_rate=1e-4, optimizer='Adam', step_size=1000, gamma=0.1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, n),
            nn.SELU(),
            nn.Linear(n, n),
            nn.SELU(),
            nn.Linear(n, n),
            nn.SELU(),
            nn.Linear(n, output_size)
        ).to(device)

        if optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        self.loss_fn = nn.MSELoss(reduction='sum')

        self.isDCT = isDCT
        if isDCT:
            self.input_name = 'coefficients-dct'
            self.output_name = 'output-dct'
        else:
            self.input_name = 'coefficients'
            self.output_name = 'output'

        self.device = device

    def forward(self, x):
        return self.model(x)

def train(models, train_dataloader, validation_dataloader, epochs=1000, stopping_loss=0.068):
    for epoch in range(epochs):
        
        if len(models) == 0:
            break
        
        print('Epoch:', epoch)
        
        for model in models:
            if model.isDCT:
                name = 'DCT'
            else:
                name = 'Non-DCT'
            
            start_time = time.time()

            train_loss = train_model_epoch(epoch, name, model, train_dataloader, backprop=True)
            validation_loss = train_model_epoch(epoch, name, model, validation_dataloader, backprop=False)

            if np.isnan(train_loss) or np.isnan(validation_loss):
                print('Loss is NaN... Removing model from training')
                models.remove(model)

            end_time = time.time()

            save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model.optimizer.state_dict(),
                'loss': train_loss
            }, 'nn-models/'+ name +'-model')

            if train_loss < stopping_loss:
                print('COMPLETED', name)
                models.remove(model)
            else:
                print(' ', name, 'Training Loss:', train_loss, 'Validation Loss:', validation_loss, 'Epoch Duration:', end_time - start_time)

    print('COMPLETED TRAINING OF MODEL')

def train_model_epoch(epoch, name, model, dataloader, backprop):
    total_loss = 0

    for ix, sample in enumerate(dataloader):
        X = sample[model.input_name].float().to(model.device)
        y = sample[model.output_name].float().to(model.device)

        if backprop:
            model.optimizer.zero_grad()
        
        output = model(X)

        loss = model.loss_fn(output, y)

        if ix == 0:
            if model.isDCT:
                y = y.cpu().detach().numpy().reshape((-1, 2))
                output = output.cpu().detach().numpy().reshape((-1, 2))

                y = idct2(y).reshape(-1)
                output = idct2(output).reshape(-1)
            else:
                y = y.cpu().detach().numpy().reshape(-1)
                output = output.cpu().detach().numpy().reshape(-1)

            plot_output(output, 
                        y, 
                        epoch, 
                        np.round(loss.item(), 4), 
                        'figures/' + name + '/')
        
        if backprop:
            loss.backward()
            model.optimizer.step()

        total_loss += loss.item()     
    
    return total_loss / len(dataloader) 