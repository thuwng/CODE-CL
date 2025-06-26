import torch
from torch import nn
from ..conceptor_operations import *

__all__ = ["CustomConv2d", "CustomLinear"]

class CustomLinear(nn.Linear):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = False,
                 n_classes=None,
                 n_experiences=None,
                 mode="IN",
                 last_layer=False,
                 threshold=0.95,
                 aperture=3,
                 weight_normalization=False,
                 num_free_dim=5
                 ):
        super(CustomLinear, self).__init__(in_features, out_features, bias)
        self.in_features = in_features
        self.n_experiences = n_experiences
        self.n_classes = n_classes
        self.basis_in = 0 * torch.eye(in_features).cuda()
        self.basis_out = 0 * torch.eye(out_features).cuda()
        self.mode = mode
        self.previous_weights = 0
        self.last_layer = last_layer
        self.threshold = threshold
        self.aperture = aperture
        self.operation_region = 1
        self.num_free_dim = num_free_dim

        self.scale = torch.nn.ModuleList()
        for t in range(20):
            self.scale.append(torch.nn.Linear(num_free_dim, num_free_dim, bias=False))

        self.conceptor_intersection = {}

        self.index = 0

        self.eps = 1e-4
        self.gain = None
        self.weight_normalization = weight_normalization

    def measure_tasks_similarity(self, task_id, conceptor):
        ratio = measure_conceptor_capacity(and_operation(self.basis_in, conceptor))
        ratio = ratio / measure_conceptor_capacity(self.basis_in)
        if ratio > 0.5 and self.in_features > 50 and self.num_free_dim > 0:
            C_intersection = and_operation(self.basis_in, conceptor)
            U, S, _ = torch.svd(C_intersection)
            self.conceptor_intersection[task_id] = [U[:, :self.num_free_dim], self.index, int(self.index + self.num_free_dim)]
            self.index = int(self.index + self.num_free_dim)

    def update_basis(self, basis_in):
        self.basis_in = basis_in

    def update_gradient(self):
        self.weight.grad.data = self.weight.grad.data - torch.matmul(self.weight.grad.data, self.basis_in)

    def get_weight(self):
        if self.weight_normalization:
            fan_in = torch.prod(torch.tensor(self.weight.shape[1:]))
            mean = torch.mean(self.weight, axis=[1], keepdims=True)
            var = torch.var(self.weight, axis=[1], keepdims=True)
            weight = (self.weight - mean) / ((var * fan_in + self.eps) ** 0.5)
            if self.gain is not None:
                weight = weight * self.gain
        else:
            weight = self.weight
        return weight

    def forward(self, input: torch.Tensor, task_id=None):
        proj_weights = 0
        if task_id in self.conceptor_intersection.keys():
            Uw, idx_low, idx_high = self.conceptor_intersection[task_id]
            proj_weights = torch.matmul(self.weight, torch.matmul(Uw, self.scale[task_id](Uw).T))
            proj_weights = proj_weights - torch.matmul(self.weight.detach(), torch.matmul(Uw, Uw.T))
        out = nn.functional.linear(input, self.get_weight() + proj_weights, self.bias)
        return out



class CustomConv2d(nn.Conv2d):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias: bool = False,
                 padding_mode: str = 'zeros',
                 n_classes=None,
                 n_experiences=None,
                 mode="IN",
                 last_layer=False,
                 device=None,
                 dtype=None,
                 threshold=0.95,
                 aperture=3,
                 weight_normalization=False,
                 num_free_dim=0
                 ):
        super(CustomConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                                           padding_mode, device, None)
        self.n_experiences = n_experiences
        self.n_classes = n_classes
        self.basis_in = 0 * torch.eye(in_channels*kernel_size*kernel_size).cuda()
        self.basis_out = 0 * torch.eye(out_channels).cuda()
        self.mode = mode
        self.previous_weights = 0
        self.last_layer = last_layer
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.aperture = aperture
        self.operation_region = 1
        self.num_free_dim = num_free_dim
        self.scale = torch.nn.ModuleList()
        for t in range(20):
            self.scale.append(torch.nn.Linear(num_free_dim, num_free_dim, bias=False))

        self.conceptor_intersection = {}
        self.index = 0

        self.gain = None
        self.eps = 1e-4
        self.weight_normalization = weight_normalization

    def measure_tasks_similarity(self, task_id, conceptor):
        ratio = measure_conceptor_capacity(and_operation(self.basis_in, conceptor))
        ratio = ratio / measure_conceptor_capacity(self.basis_in)
        if ratio > 0.5 and (self.kernel_size**2)*self.in_channels > self.num_free_dim and self.num_free_dim > 0:
            print(f"Layer {conceptor.size(0)} in Region 2")
            self.operation_region = 2
            C_intersection = and_operation(self.basis_in, conceptor)
            U, S, _ = torch.svd(C_intersection)
            self.conceptor_intersection[task_id] = [U[:, :self.num_free_dim], self.index, int(self.index + self.num_free_dim)]
            self.index = int(self.index + self.num_free_dim)
            print(self.conceptor_intersection.keys())

    def update_basis(self, basis_in):
        self.basis_in = basis_in

    def update_gradient(self):
        self.weight.grad.data = self.weight.grad.data - torch.matmul(self.weight.grad.data.view(self.weight.grad.data.size(0), -1), self.basis_in).view_as(self.weight)

    def get_weight(self):
        if self.weight_normalization:
            fan_in = torch.prod(torch.tensor(self.weight.shape[1:]))
            mean = torch.mean(self.weight, axis=[1, 2, 3], keepdims=True)
            var = torch.var(self.weight, axis=[1, 2, 3], keepdims=True)
            weight = (self.weight - mean) / ((var * fan_in + self.eps) ** 0.5)
            if self.gain is not None:
                weight = weight * self.gain
        else:
            weight = self.weight
        return weight

    def forward(self, input: torch.Tensor, task_id=None):
        out_channels = self.weight.size(0)
        proj_weights = 0
        if task_id in self.conceptor_intersection.keys():
            Uw, idx_low, idx_high = self.conceptor_intersection[task_id]
            proj_weights = torch.matmul(self.weight.view(out_channels, -1), torch.matmul(Uw, self.scale[task_id](Uw).T)).view_as(self.weight)
            proj_weights = proj_weights - torch.matmul(self.weight.view(out_channels, -1), torch.matmul(Uw, Uw.T)).view_as(self.weight)

        out = nn.functional.conv2d(input, self.get_weight() + proj_weights, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out