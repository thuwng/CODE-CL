from models.conceptor_operations import *
import torch
import logging

__all__ = ["update_basis"]


def update_basis(mat_list, threshold=[0.95, 0.99, 0.99], conceptor_list=[], aperture=[8, 8, 8, 16, 16, 16],
                 model_name="AlexNet", memory_threshold=0.95, lower_sval_bound=0.2, print_logs=True):
    if model_name == 'ResNet18':
        kernel_list = [3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 1, 3, 3, 3, 3, 1, 3, 3, 3]
        stride_list = [1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1, 1]
        padding_list = [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1]
    elif model_name == 'AlexNet':
        kernel_list = [4, 3, 2]
        stride_list = [1, 1, 1]
        padding_list = [0, 0, 0]
    elif model_name == 'MLP':
        pass
    else:
        raise NotImplementedError(f"{model_name} has not been implemented")
    if not conceptor_list:
        # After First Task
        for i in range(len(mat_list)):
            if len(mat_list[i].size()) == 4:
                data_x = mat_list[i].cuda()
                x_unf = torch.nn.functional.unfold(data_x,
                                                   kernel_size=(kernel_list[i], kernel_list[i]),
                                                   stride=(stride_list[i], stride_list[i]),
                                                   padding=(padding_list[i], padding_list[i]))
                x_unf = x_unf.permute(0, 2, 1).contiguous()
                activation = x_unf.view(-1, x_unf.size(2)).T
            else:
                activation = mat_list[i].cuda().T
            flag = True
            gain = 1.1
            C = compute_conceptor(activation, aperture=aperture[i])
            while flag:
                ratio = torch.norm(torch.matmul(C, activation), dim=0).sum() / torch.norm(activation,
                                                                                          dim=0).sum()
                if ratio < memory_threshold:
                    C = aperture_adaptation(C, gain)
                else:
                    flag = False
            if lower_sval_bound > 0:
                UC, SC, UtC = torch.svd(C)
                SC[SC < lower_sval_bound] = 0
                C = torch.matmul(UC, torch.matmul(torch.diag(SC), UC.T))
            conceptor_list.append(C.cuda())
    else:
        for i in range(len(mat_list)):
            if len(mat_list[i].size()) == 4:
                data_x = mat_list[i].cuda()
                x_unf = torch.nn.functional.unfold(data_x,
                                                   kernel_size=(kernel_list[i], kernel_list[i]),
                                                   stride=(stride_list[i], stride_list[i]),
                                                   padding=(padding_list[i], padding_list[i]))
                x_unf = x_unf.permute(0, 2, 1).contiguous()
                activation = x_unf.view(-1, x_unf.size(2)).T
            else:
                activation = mat_list[i].cuda().T
            C = conceptor_list[i]
            flag = True
            gain = 1.1
            B = compute_conceptor(activation, aperture=aperture[i])
            while flag:
                ratio = torch.norm(torch.matmul(B, activation), dim=0).sum() / torch.norm(activation,
                                                                                          dim=0).sum()
                if ratio < memory_threshold:
                    B = aperture_adaptation(B, gain)
                else:
                    flag = False
            C = or_operation(C, B, aperture=aperture[i])
            if lower_sval_bound > 0:
                UC, SC, UtC = torch.svd(C)
                SC[SC < lower_sval_bound] = 0
                C = torch.matmul(UC, torch.matmul(torch.diag(SC), UC.T))
            conceptor_list[i] = C

    if print_logs:
        logging.info('-' * 40)
        logging.info('Conceptors Summary')
        logging.info('-' * 40)
        for i in range(len(conceptor_list)):
            S_conceptor = torch.linalg.svdvals(conceptor_list[i])
            logging.info('Layer {} : {:.3f}% \t max_val = {:.3f} \t min_val = {:.3f} \t # directions = {:.3f}/{}'.format(
                i + 1,
                100 * measure_conceptor_capacity(conceptor_list[i]),
                torch.max(S_conceptor), torch.min(S_conceptor), torch.sum(S_conceptor>1e-4), len(S_conceptor)))
        logging.info('-' * 40)
    return conceptor_list