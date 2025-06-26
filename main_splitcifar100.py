import torch
from torch import nn
import numpy as np
import mlflow
import logging
import warnings
from models.nn_models import AlexNet
from torch.utils.data.dataset import TensorDataset
from avalanche.benchmarks.utils import AvalancheDataset
from cl_method import code_cl, strategy
from utils import parse_args
import os
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == '__main__':
    args = parse_args()

    folder_name = f"{args.experiment_name}_{torch.randint(low=0, high=100000, size=[1]).item()}"
    args.save_path = args.save_path + "/" + folder_name
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Log initialization
    log_path = args.save_path + "/log.log"
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S',
                        filename=log_path,
                        filemode="a")
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(args)
    logging.info('=> Everything will be saved to {}'.format(log_path))

    experiment_name = args.experiment_name
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        n_experiences = args.n_experiences
        epochs = args.epochs
        lr = args.lr
        threshold_conv = args.threshold_conv
        threshold_linear = args.threshold_linear
        batch_size = args.batch_size
        aperture = args.aperture
        dataset = args.dataset
        model = args.model
        dropout = args.dropout
        data_aug = args.data_aug
        basis_bs = args.basis_batch_size
        avg_pool = args.avg_pool
        print_freq = args.print_freq
        aperture_gain = args.aperture_gain

        for key in args.__dict__.keys():
            mlflow.log_param(key, args.__dict__[key])


        from utils import cifar100 as cf100
        data, taskcla, inputsize = cf100.get(pc_valid=0.05)

        if model == "AlexNet":
            model_ = AlexNet(n_experiences=n_experiences, n_classes=int(100//n_experiences),
                             threshold_conv=threshold_conv, threshold_linear=threshold_linear,
                             num_free_dim=args.num_free_dim).cuda()
        else:
            raise NotImplementedError(f"{model} is not available")

        cl_strategy = strategy.MyStrategy(model_, None, nn.CrossEntropyLoss(), epochs=epochs,
                                          batch_size=batch_size, threshold=[0, 0],
                                          aperture=aperture, dropout=dropout, data_aug=data_aug,
                                          basis_bs=basis_bs, avg_pool=avg_pool, transform_test=None,
                                          transform_train=None, dataset_name=dataset, model_name=model,
                                          print_freq=print_freq, aperture_gain=aperture_gain, patience=args.patience,
                                          lr_decay=args.lr_decay, lr_threshold=args.lr_threshold)

        # Training Loop
        accuracy_history = []
        accuracy_list = []
        logging.info('Starting experiment...')

        xtest = data[0]['test']['x']
        ytest = data[0]['test']['y']

        torch_data_test = TensorDataset(xtest, ytest)
        experience_test_zero = AvalancheDataset(torch_data_test)

        task_id = 0
        task_list = []
        for k, ncla in taskcla:
            # specify threshold hyperparameter

            logging.info('*' * 100)
            logging.info('Task {:2d} ({:s})'.format(k, data[k]['name']))
            logging.info('*' * 100)
            xtrain = data[k]['train']['x']
            ytrain = data[k]['train']['y']
            xvalid = data[k]['valid']['x']
            yvalid = data[k]['valid']['y']
            xtest = data[k]['test']['x']
            ytest = data[k]['test']['y']
            task_list.append(k)

            torch_data_train = TensorDataset(xtrain, ytrain)
            experience_train = AvalancheDataset(torch_data_train)

            torch_data_test = TensorDataset(xvalid, yvalid)
            experience_test = AvalancheDataset(torch_data_test)

            logging.info("Start of experience {0}".format(task_id))
            # cl_strategy.similarity_analysis(benchmark.train_stream[exp_id])
            cl_strategy.train(experience_train, experience_test, experience_zero=experience_test_zero)

            for exp_test_id in task_list:
                xeval = data[exp_test_id]['test']['x']
                yeval = data[exp_test_id]['test']['y']
                torch_data_eval = TensorDataset(xeval, yeval)
                experience_eval = AvalancheDataset(torch_data_eval)
                logging.info('Testing experience: {0}'.format(exp_test_id))
                if exp_test_id <= task_id:
                    acc = cl_strategy.eval(experience_eval, task_id, exp_test_id)
                    accuracy_list.append(acc)
            logging.info(cl_strategy.forgetting_metric.result())
            fogetting_dict = cl_strategy.forgetting_metric.result()
            for key in fogetting_dict.keys():
                mlflow.log_metric("Forgetting_" + key, fogetting_dict[key], step=task_id)
            if task_id + 1 < n_experiences:
                cl_strategy.update_basis(experience_train, task_id)
            accuracy_history.append(accuracy_list)
            accuracy_list = []
            avg_forgetting = strategy.average_forgetting_metric(fogetting_dict)
            logging.info(f"Average Forgetting: {avg_forgetting}")
            logging.info(f"Average Accuracy: {np.mean(np.array(accuracy_history[-1]))}")
            mlflow.log_metric("avg_forgetting", avg_forgetting, step=task_id)
            mlflow.log_metric("avg_accuracy", np.mean(np.array(accuracy_history[-1])), step=task_id)

            task_id += 1
            # break

        avg_forgetting = strategy.average_forgetting_metric(fogetting_dict)
        mlflow.log_metric("avg_forgetting", avg_forgetting)
        mlflow.log_metric("avg_accuracy", np.mean(np.array(accuracy_history[-1])))
