import torch
from .basemodel import BaseModel
from .layers import CustomLinear

__all__ = ['MLP']


class MLP(BaseModel):
    def __init__(self, threshold_linear=0.95, num_free_dim=5):
        super(MLP, self).__init__()
        hidden_size = 100
        mode = "IN"

        self.fc1 = CustomLinear(28 * 28, 100, bias=False, mode=mode, threshold=threshold_linear, num_free_dim=num_free_dim)
        self.fc2 = CustomLinear(100, 100, bias=False, mode=mode, threshold=threshold_linear, num_free_dim=num_free_dim)
        self.fc3 = CustomLinear(100, 10, bias=False, mode=mode, threshold=threshold_linear, last_layer=True, num_free_dim=num_free_dim)

    def measure_tasks_similarity(self, task_id, conceptor_list):
        self.fc1.measure_tasks_similarity(task_id, conceptor_list[0])
        self.fc2.measure_tasks_similarity(task_id, conceptor_list[1])
        self.fc3.measure_tasks_similarity(task_id, conceptor_list[2])

    def update_basis(self, conceptor_list:list):
        self.fc1.update_basis(basis_in=conceptor_list[0])
        self.fc2.update_basis(basis_in=conceptor_list[1])
        self.fc3.update_basis(basis_in=conceptor_list[2])


    def update_gradient(self):
        self.fc1.update_gradient()
        self.fc2.update_gradient()
        self.fc3.update_gradient()

    def forward(self, x, labels=None, experience_id=None):
        x = x.view(x.shape[0], -1)
        x = torch.relu(self.fc1(x, task_id=experience_id))
        x = torch.relu(self.fc2(x, task_id=experience_id))
        y_pred = self.fc3(x, task_id=experience_id)
        return y_pred

    def forward_all_layers(self, x, experience_id=None):
        x0 = x.view(x.shape[0], -1)
        x1 = torch.relu(self.fc1(x0, task_id=experience_id))
        x2 = torch.relu(self.fc2(x1, task_id=experience_id))
        return x0, x1, x2
