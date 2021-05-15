import torch 
from torch import nn, optim
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, dropout=0.5):
        
        super().__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        
        x = x.view(x.shape[0], -1)
        
        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
            
        x = F.log_softmax(self.output(x), dim=1)
        
        return x
    
def train(model, trainloader, testloader, criterion, optimizer, epochs=5, print_every=40):
    
    train_losses = 0
    test_losses = 0
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            
            output = model(images)
            loss = criterion(output, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        else:
            test_loss = 0
            accuracy = 0
            with torch.no_grad():
                model.eval()
                
                # Validation
                for images, labels in testloader:
                    
                    log_ps = model(images)
                    test_loss += criterion(log_ps, labels).item()
                    ps = torch.exp(log_ps)
                    
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))
                    
                print(
                    "Epochs: {}/{}".format(e+1, epochs),
                    "Training Loss {:.3f}".format(running_loss/len(trainloader)),
                    "Validation Loss {:.3f}".format(test_loss/len(testloader)),
                    "Accuracy {:.3f}%".format((accuracy/len(testloader)) * 100)
                    )
                
                
                model.train()           
            
        
        