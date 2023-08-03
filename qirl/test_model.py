import torch


class TestModel(torch.nn.Module):
    def __init__(self, n_input_features, n_outputs):
        super().__init__()
        self.Dense1 = torch.nn.Linear(n_input_features, 16)
        self.Dense2 = torch.nn.Linear(16, 16)
        self.Output = torch.nn.Linear(16, n_outputs)
        self.model = torch.nn.Sequential(
            self.Dense1,
            torch.nn.ELU(),
            self.Dense2,
            torch.nn.ELU(),
            self.Output,
            torch.nn.Softmax()
        )
    
    def forward(self, x):
        return torch.tensor(self.model(x), dtype=torch.float32, requires_grad=True)