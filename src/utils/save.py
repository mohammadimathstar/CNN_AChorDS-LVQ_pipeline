"""
Modified Script - Original Source: https://github.com/M-Nauta/ProtoTree/tree/main

Description:
This script is a modified version of the original script by M. Nauta.
Modifications have been made to suit specific requirements or preferences.

"""

import argparse
from src.AChorDSLVQ.model import Model


def save_model(
        model: Model,
        epoch: int,
        config: dict,
        # log: Log,
        # args: argparse.Namespace
):
    model.eval()
    # Save latest model
    model.save(f'{config['train']['checkpoint_dir']}/latest')
    model.save_state(f'{config['train']['checkpoint_dir']}/latest')
    # model.save(f'{log.checkpoint_dir}/latest')
    # model.save_state(f'{log.checkpoint_dir}/latest')

    # Save model every 10 epochs
    if epoch == config['train']['num_epochs'] or epoch%10==0:
        model.save(f'{config['train']['checkpoint_dir']}/epoch_{epoch}')
        model.save_state(f'{config['train']['checkpoint_dir']}/epoch_{epoch}')

def save_best_train_model(
        model: Model,
        best_train_acc: float,
        train_acc: float,
        config: dict,
):
    model.eval()
    if train_acc > best_train_acc:
        best_train_acc = train_acc
        model.save(f'{config['train']['checkpoint_dir']}/best_train_model')
        model.save_state(f'{config['train']['checkpoint_dir']}/best_train_model')

    return best_train_acc

def save_best_test_model(
        model: Model,
        best_test_acc: float,
        test_acc: float,
        config: dict,
):
    model.eval()
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        model.save(f'{config['train']['checkpoint_dir']}/best_test_model')
        model.save_state(f'{config['train']['checkpoint_dir']}/best_test_model')
    return best_test_acc

def save_model_description(
        model: Model,
        description: str,
        config: dict,
):
    model.eval()
    # Save model with description
    model.save(f'{config['train']['checkpoint_dir']}/'+description)
    model.save_state(f'{config['train']['checkpoint_dir']}/'+description)
