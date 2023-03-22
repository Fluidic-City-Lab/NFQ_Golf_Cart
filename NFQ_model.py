import torch 
import torch.nn as nn 

class NFQNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 8),
            nn.Sigmoid(),
            nn.Linear(8, 5),
            nn.Sigmoid(),
            nn.Linear(5, 1),
            nn.Sigmoid(), # sigmoid final layer implies costs must be 0 < C < 1!
        )

        def init_weights(m):
            
            if type(m) == nn.Linear:
                # uniform according to NFQ paper
                nn.init.uniform_(m.weight, -0.5, 0.5)

        self.layers.apply(init_weights)

    def forward(self, x):
        return self.layers(x)
        

# Use case:
# Print model structure and weights, correctly initialized?
# model = NFQNetwork()
# print(model)
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)
