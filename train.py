import yaml
import torch

from src.stages.data import get_dataloaders
from src.stages.create_network import create_network
from src.stages.train_epoch import train_epoch
from src.utils.glvq import get_loss_function
from src.features.backbone_net import freeze
from src.utils.logs import get_logger
from src.stages.test import eval
from src.utils.save import save_model, save_best_train_model, save_best_test_model


def train(config_file_path: str):

    with open(config_file_path, "r") as file:
        config = yaml.safe_load(file)

    # Save the initial network
    with open(f"{config['train']['checkpoint_dir']}/params.yaml", 'w') as f:
        yaml.dump(config, f)

    if config['base']['use_cuda'] and torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
    else:
        device = torch.device('cpu')

    logger = get_logger('MAIN',
                        log_level=config['base']['log_level'],
                        log_file=config['base']['log_file'])

    # Prepare data loaders
    trainloader, testloader, classes, shape = get_dataloaders(config_file_path)

    # Create a convolutional network
    model, optimizer_net, optimizer_proto, optimizer_rel, params_to_freeze, params_to_train = create_network(config_file_path, device)

    
    # Save meta-data
    model.save_state(f"{config['train']['checkpoint_dir']}/model_init")

    loss = get_loss_function(config_file_path)
    logger.info(f"The loss function of type '{config['train']['hyperparams']['act_fun']}' has been created.")

    best_train_acc = 0
    best_test_acc = 0

    epoch = 1

    # Train the model
    for epoch in range(epoch, config['train']['num_epochs'] + 1):
        
        logger.info("Epoch %i" % epoch)

        # freeze part of network for some epochs if indicated in args
        freeze(model, 
               epoch, 
               params_to_freeze, 
            #    params_to_train,
               config,
               )#, args, log)

        # Train model
        train_info = train_epoch(
            model,
            trainloader,
            epoch,
            loss,
            # args,
            optimizer_net,
            optimizer_proto,
            optimizer_rel,
            device,
            config,
            # log_prefix=log_prefix,
            progress_prefix='Train Epoch',
        )

        # save
        save_model(model, epoch, config)

        # complete the following
        best_train_acc = save_best_train_model(model, best_train_acc, train_info['train_accuracy'], config)

        eval_info = eval(model, testloader, epoch, loss, device, config)
        original_test_acc = eval_info['test_accuracy']
        best_test_acc = save_best_test_model(model,
                                             best_test_acc,
                                             eval_info['test_accuracy'],
                                             config)
        # log.log_values('log_epoch_overview', epoch, eval_info['test_accuracy'], train_info['train_accuracy'],
        #                train_info['loss'])


if __name__ == '__main__':
    train('params.yaml')