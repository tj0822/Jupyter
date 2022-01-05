"""Trainer 클래스 정의
"""

import os

import torch
from sklearn.metrics import f1_score


class Trainer():
    """ Trainer
        epoch에 대한 학습 및 검증 절차 정의
    
    Attributes:
        model (`model`)
        device (str)
        loss_fn (Callable)
        metric_fn (Callable)
        optimizer (`optimizer`)
        scheduler (`scheduler`)
    """

    def __init__(self, model, device, loss_fn, metric_fn, optimizer=None, scheduler=None, logger=None):
        """ 초기화
        """
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger

        # History

    def train_epoch(self, dataloader, epoch_index=0):
        """ 한 epoch에서 수행되는 학습 절차

        Args:
            dataloader (`dataloader`)
            epoch_index (int)
        """
        self.model.train()
        self.train_total_loss = 0
        target_lst = []
        pred_lst = []
        for batch_index, (data, target) in enumerate(dataloader):
            data = data.to(self.device)
            target = target.to(self.device)
            output = self.model(data)
            self.optimizer.zero_grad()
            loss = self.loss_fn(output, target)
            self.train_total_loss += loss
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            target_lst.extend(target.tolist())
            pred_lst.extend(output.argmax(dim=1).tolist())
        self.train_mean_loss = self.train_total_loss / len(dataloader)
        self.train_score = f1_score(y_true=target_lst, y_pred=pred_lst, average='macro')
        msg = f'Epoch {epoch_index}, Train, loss: {self.train_mean_loss}, Score: {self.train_score}'
        print(msg)
        self.logger.info(msg) if self.logger else print(msg)


    def validate_epoch(self, dataloader, epoch_index=0):
        """ 한 epoch에서 수행되는 검증 절차

        Args:
            dataloader (`dataloader`)
            epoch_index (int)
        """
        self.model.eval()
        self.val_total_loss = 0
        target_lst = []
        pred_lst = []
        with torch.no_grad():
            for batch_index, (data, target) in enumerate(dataloader):
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.model(data)
                loss = self.loss_fn(output, target)
                self.val_total_loss += loss
                target_lst.extend(target.tolist())
                pred_lst.extend(output.argmax(dim=1).tolist())
            self.val_mean_loss = self.val_total_loss / len(dataloader)
            self.validation_score = f1_score(y_true=target_lst, y_pred=pred_lst, average='macro')
            msg = f'Epoch {epoch_index}, Validation, loss: {self.val_mean_loss}, Score: {self.validation_score}'
            print(msg)
            self.logger.info(msg) if self.logger else print(msg)

