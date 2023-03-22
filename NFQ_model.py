import torch 
import torch.nn as nn 

class NFQNetwork(nn.Module):
    """
    Neural Network size was varied in 5 sizes each with param counts: 
    Param counts: 39, 61, 91, 121, 171
    Example calaulation: 91: [4, 8, 5, 1], calculated as 4*8 + 8 + 8*5 + 5 + 5*1 + 1
    """
    def __init__(self, param_count):
        super().__init__()
        self.param_count = param_count
        self.layers = self.create_layers()

        def init_weights(m):
            
            if type(m) == nn.Linear:
                # uniform according to NFQ paper
                nn.init.uniform_(m.weight, -0.5, 0.5)

        self.layers.apply(init_weights)

    def create_layers(self):
        if self.param_count == 39:
            config = [4, 4, 3, 1]
        elif self.param_count == 61:
            config = [4, 5, 5, 1]
        elif self.param_count == 91:
            config = [4, 8, 5, 1]
        elif self.param_count == 121:
            config = [4, 8, 8, 1]
        elif self.param_count == 171:
            config = [4, 10, 10, 1]
        else:
            raise ValueError("Invalid parameter count. Choose from: 39, 61, 91, 121, 171")

        layers = []
        for i in range(len(config) - 1):
            layers.append(nn.Linear(config[i], config[i + 1]))
            layers.append(nn.Sigmoid()) # sigmoid final layer implies costs must be 0 < C < 1!

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def count_parameters(self):
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return count
        

# Use case:
# Print model structure and weights, correctly initialized?
# model = NFQNetwork(171)
# print(model)
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)

# print(f"\nTrainable parameters: {model.count_parameters()}")

