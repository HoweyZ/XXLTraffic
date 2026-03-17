from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from models.CAMEL import NLLLoss

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.nll_criterion = NLLLoss()

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def _prepare_marks(self, batch_x_mark, batch_y_mark):
        if self.args.model == 'CAMEL':
            return batch_x_mark, batch_y_mark
        model_x_mark = batch_x_mark[..., :4] if batch_x_mark is not None and batch_x_mark.shape[-1] > 4 else batch_x_mark
        model_y_mark = batch_y_mark[..., :4] if batch_y_mark is not None and batch_y_mark.shape[-1] > 4 else batch_y_mark
        return model_x_mark, model_y_mark

    def _unwrap_model_output(self, raw_out):
        if isinstance(raw_out, tuple):
            return raw_out[0], raw_out[1]
        if isinstance(raw_out, dict):
            return raw_out.get('pred'), raw_out
        return raw_out, {}

    def _compute_total_loss(self, batch_x, dec_inp, batch_y, batch_x_mark, batch_y_mark, criterion):
        model_x_mark, model_y_mark = self._prepare_marks(batch_x_mark, batch_y_mark)

        raw_out = self.model(batch_x, model_x_mark, dec_inp, model_y_mark)
        outputs, extra = self._unwrap_model_output(raw_out)

        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        target = batch_y[:, -self.args.pred_len:, f_dim:]

        task_loss = criterion(outputs, target)
        if self.args.model == 'CAMEL' and self.args.camel_use_nll and extra.get('sigma') is not None:
            sigma = extra['sigma'][:, f_dim:, :]
            task_loss = self.nll_criterion(outputs.transpose(1, 2), target.transpose(1, 2), sigma)

        total_loss = task_loss
        aux_losses = extra.get('aux_losses', {}) if isinstance(extra, dict) else {}
        if self.args.model == 'CAMEL':
            total_loss = total_loss + self.args.lambda_mem * aux_losses.get('mem', total_loss.new_tensor(0.0))
            total_loss = total_loss + self.args.lambda_ode * aux_losses.get('ode', total_loss.new_tensor(0.0))
            total_loss = total_loss + self.args.lambda_smooth * aux_losses.get('smooth', total_loss.new_tensor(0.0))

        return total_loss, outputs, target

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).to(self.device)

                loss, _, _ = self._compute_total_loss(batch_x, dec_inp, batch_y, batch_x_mark, batch_y_mark, criterion)
                total_loss.append(loss.item())

        self.model.train()
        return np.average(total_loss)

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        loss, _, _ = self._compute_total_loss(batch_x, dec_inp, batch_y, batch_x_mark, batch_y_mark, criterion)
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss, _, _ = self._compute_total_loss(batch_x, dec_inp, batch_y, batch_x_mark, batch_y_mark, criterion)
                    loss.backward()
                    model_optim.step()

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
                    iter_count = 0
                    time_now = time.time()

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).to(self.device)

                model_x_mark, model_y_mark = self._prepare_marks(batch_x_mark, batch_y_mark)
                raw_out = self.model(batch_x, model_x_mark, dec_inp, model_y_mark)
                outputs, _ = self._unwrap_model_output(raw_out)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                preds.append(outputs)
                trues.append(batch_y)

        preds = np.array(preds).reshape(-1, self.args.pred_len, preds[0].shape[-1])
        trues = np.array(trues).reshape(-1, self.args.pred_len, trues[0].shape[-1])

        folder_path = './results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        with open('result_long_term_forecast.txt', 'a') as f:
            f.write(setting + '  \n')
            f.write('mse:{}, mae:{}'.format(mse, mae))
            f.write('\n\n')

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        return
