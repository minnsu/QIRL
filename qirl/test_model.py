import torch
import torch.nn.functional as F

class TestModel(torch.nn.Module):
    def __init__(self, n_input_features, n_outputs):
        super().__init__()
        self.Dense1 = torch.nn.Linear(n_input_features, 32)
        self.Dense2 = torch.nn.Linear(32, 64)
        self.Dense3 = torch.nn.Linear(64, 128)
        self.Dense4 = torch.nn.Linear(128, 128)
        self.Dense5 = torch.nn.Linear(128, 64)
        self.Output = torch.nn.Linear(64, n_outputs)
        self.model = torch.nn.Sequential(
            self.Dense1,
            torch.nn.Sigmoid(),
            self.Dense2,
            self.Dense3,
            self.Dense4,
            torch.nn.Sigmoid(),
            self.Dense5,
            torch.nn.Sigmoid(),
            self.Output,
            torch.nn.Softmax()
        )
    
    def forward(self, x):
        return torch.tensor(self.model(x), dtype=torch.float32, requires_grad=True)

class ActorCriticModel(torch.nn.Module):
    def __init__(self, n_features, n_policy_output):
        super().__init__()
        self.affine1 = torch.nn.Linear(n_features, 64)        
        self.affine2 = torch.nn.Linear(64, 128)        
        self.affine3 = torch.nn.Linear(128, 64)        

        self.act_prob = torch.nn.Linear(64, n_policy_output)
        self.q_value = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        x = F.sigmoid(self.affine2(x))
        x = self.affine3(x)
        act_prob = F.softmax(self.act_prob(x), dim=-1)
        q_value = self.q_value(x)
        return act_prob, q_value