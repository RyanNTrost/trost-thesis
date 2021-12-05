from torch import nn, optim, save
import time
import numpy as np
from visualization import plot_output, plot_error
from laplace_dataset import idct2

class LaplaceModel(nn.Module):
    def __init__(self, n, num_layers, input_size, output_size, isDCT, device, learning_rate=1e-4, optimizer='Adam', step_size=1000, gamma=0.1):
        super().__init__()
        modules = [nn.Linear(input_size, n), nn.SELU()]
        for i in range(num_layers):
            modules.append(nn.Linear(n, n))
            modules.append(nn.SELU())
        modules.append(nn.Linear(n, output_size))
        self.model = nn.Sequential(*modules).to(device)

        self.input_size = input_size
        self.output_size_sqrt = int(np.sqrt(output_size))

        if optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        self.loss_fn = nn.MSELoss(reduction='sum')
        # self.loss_fn = nn.MSELoss()

        self.isDCT = isDCT
        if isDCT:
            self.input_name = 'coefficients-dct'
            self.output_name = 'output-dct'
            self.name = 'DCT'
        else:
            self.input_name = 'coefficients'
            self.output_name = 'output'
            self.name = 'Non-DCT'

        self.device = device

    def forward(self, x):
        return self.model(x)

def train(models, train_dataloader, validation_dataloader, epochs=1000, stopping_loss=0.068):
    training_losses = {}
    validation_losses = {}
    
    for model in models:
        training_losses[model.name] = []
        validation_losses[model.name] = []
    
    for epoch in range(epochs):
        if len(models) == 0:
            break
        
        print('Epoch:', epoch)
        
        for model in models:
            start_time = time.time()

            train_loss = train_model_epoch(epoch, model.name, model, train_dataloader, backprop=True)
            validation_loss = train_model_epoch(epoch, model.name, model, validation_dataloader, backprop=False)

            training_losses[model.name].append(train_loss)
            validation_losses[model.name].append(validation_loss)

            if np.isnan(train_loss) or np.isnan(validation_loss):
                print('Loss is NaN... Removing model from training')
                models.remove(model)

            end_time = time.time()

            # save(model, 'nn-models/'+ model.name +'-model')

            # save({
            #     'epoch': epoch,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': model.optimizer.state_dict(),
            #     'loss': train_loss
            # }, 'nn-models/'+ name +'-model')

            if train_loss < stopping_loss:
                print('COMPLETED', model.name)
                models.remove(model)
            else:
                print(' ', model.name, 'Training Loss:', train_loss, 'Validation Loss:', validation_loss, 'Epoch Duration:', end_time - start_time)

    print('COMPLETED TRAINING OF MODEL')

    plot_error(training_losses, 'Training Loss', 'figures/training_loss/', save=False)
    plot_error(validation_losses, 'Validation Loss', 'figures/validation_loss/', save=False)

def train_model_epoch(epoch, name, model, dataloader, backprop):
    total_loss = 0

    for ix, sample in enumerate(dataloader):
        X = sample[model.input_name].float().to(model.device)
        y = sample[model.output_name].float().to(model.device)

        if backprop:
            model.optimizer.zero_grad()
        
        output = model(X)

        loss = model.loss_fn(output, y)

        if backprop and ix == 0:
            if model.isDCT:
                y = y.cpu().detach().numpy().reshape((model.output_size_sqrt, model.output_size_sqrt))
                output = output.cpu().detach().numpy().reshape((model.output_size_sqrt, model.output_size_sqrt))

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


def plot_test(model, dataloader):
    for ix, sample in enumerate(dataloader):
        X = sample[model.input_name].float().to(model.device)
        y = sample[model.output_name].float().to(model.device)
        
        output = model(X)

        loss = model.loss_fn(output, y)

        if model.isDCT:
            y = y.cpu().detach().numpy().reshape((model.output_size_sqrt, model.output_size_sqrt))
            output = output.cpu().detach().numpy().reshape((model.output_size_sqrt, model.output_size_sqrt))

            y = idct2(y).reshape(-1)
            output = idct2(output).reshape(-1)
        else:
            y = y.cpu().detach().numpy().reshape(-1)
            output = output.cpu().detach().numpy().reshape(-1)

        plot_output(output, 
                    y, 
                    ix, 
                    np.round(loss.item(), 4), 
                    '',
                    False)
 