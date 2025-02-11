
from tqdm import tqdm

import torch
import torch.utils.data
import torch.utils.data
from torch.utils.data import DataLoader

from src.AChorDSLVQ.model import Model
from src.AChorDSLVQ.prototypes import rotate_prototypes
from src.utils.grassmann import orthogonalize_batch
from src.utils.logs import get_logger
# from src.utils.log import Log

def train_epoch(
        model: Model,
        train_loader: DataLoader,
        epoch: int,
        loss,
        # args: argparse.Namespace,
        optimizer_net: torch.optim.Optimizer,
        optimizer_protos: torch.optim.Optimizer,
        optimizer_rel: torch.optim.Optimizer,
        device,
        config: dict,
        # log: Log = None,
        # log_prefix: str = 'log_train_epochs',
        progress_prefix: str = 'Train Epoch'
) -> dict:
    
    logger = get_logger('TRAIN',
                        log_level=config['base']['log_level'],
                        log_file=config['base']['log_file'])

    model = model.to(device)

    # to store information about the procedure
    train_info = dict()
    total_loss = 0
    total_acc = 0

    # create a log
    # log_loss = f"{log_prefix}_losses"

    # to show the progress-bar
    train_iter = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=progress_prefix + ' %s' % epoch,
        ncols=0
    )

    acc_mean = 0

    # training process (one epoch)
    for i, (xtrain, ytrain) in enumerate(train_loader):
        nbatch = xtrain.shape[0]

        # ****** for the first solution
        optimizer_protos.zero_grad()
        optimizer_rel.zero_grad()
        optimizer_net.zero_grad()
        

        xtrain, ytrain = xtrain.to(device), ytrain.to(device)
        distances, Qw = model(xtrain)
        cost, iplus, iminus = loss(
            ytrain,
            model.prototype_layer.yprotos_mat,
            model.prototype_layer.yprotos_comp_mat,
            distances)

        cost.backward()

        
        with torch.no_grad():
            winners_ids, _ = torch.stack([iplus, iminus], axis=1).sort(axis=1)
            rotated_proto1, rotated_proto2 = rotate_prototypes(model.prototype_layer.xprotos, Qw, winners_ids)
            model.prototype_layer.xprotos[winners_ids[torch.arange(nbatch), 0]] = rotated_proto1
            model.prototype_layer.xprotos[winners_ids[torch.arange(nbatch), 1]] = rotated_proto2

        optimizer_protos.step()
        optimizer_rel.step()
        optimizer_net.step()

        with torch.no_grad():
            model.prototype_layer.xprotos[winners_ids[torch.arange(nbatch), 0]] = orthogonalize_batch(
                model.prototype_layer.xprotos[winners_ids[torch.arange(nbatch), 0]]
            )
            model.prototype_layer.xprotos[winners_ids[torch.arange(nbatch), 1]] = orthogonalize_batch(
                model.prototype_layer.xprotos[winners_ids[torch.arange(nbatch), 1]]
            )
            #CHECK
            LOW_BOUND_LAMBDA = 0.0001
            model.prototype_layer.relevances[0, torch.argwhere(model.prototype_layer.relevances < LOW_BOUND_LAMBDA)[:, 1]] = LOW_BOUND_LAMBDA
            model.prototype_layer.relevances[:] = model.prototype_layer.relevances[
                                                  :] / model.prototype_layer.relevances.sum()


        # compute the accuracy
        yspred = model.prototype_layer.yprotos[distances.argmin(axis=1)]
        acc = torch.sum(torch.eq(yspred, ytrain)).item() / float(len(xtrain))

        train_iter.set_postfix_str(
            f"Batch [{i + 1}/{len(train_loader)}, Loss: {cost.sum().item(): .3f}, Acc: {acc * 100: .2f}"
        )
        acc_mean += acc

        # update the total metrics
        total_acc += acc
        total_loss += torch.sum(cost).item()

        # write a log
        # if log is not None:
        #     log.log_values(log_loss, epoch, i + 1, torch.sum(cost).item(), acc)

    print('\n', model.prototype_layer.relevances)

    train_info['loss'] = total_loss / float(i + 1)
    train_info['train_accuracy'] = total_acc / float(i + 1)

    logger.info(f"Train loss: {train_info['loss']:.3f}, Train acc: {train_info['train_accuracy'] * 100:.2f}")

    return train_info

