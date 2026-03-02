from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from thop import profile, clever_format
import torch.nn.functional as F
from itertools import combinations


warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        params = self.model.parameters()
        if self.args.enable_btta and self.args.btta_stage == 'btta' and self.args.freeze_backbone_in_btta:
            for p in self.model.parameters():
                p.requires_grad = False
            params = [p for p in self.model.parameters() if p.requires_grad]
            if len(params) == 0:
                print('Warning: backbone is frozen in BTTA stage and no adapters are registered; falling back to full-model update.')
                for p in self.model.parameters():
                    p.requires_grad = True
                params = self.model.parameters()

        model_optim = optim.Adam(params, lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _extract_trend(self, x):
        kernel = max(1, int(self.args.trend_kernel))
        if kernel % 2 == 0:
            kernel += 1
        x_reshape = x.transpose(1, 2)
        trend = F.avg_pool1d(x_reshape, kernel_size=kernel, stride=1, padding=kernel // 2)
        return trend.transpose(1, 2)

    def _stable_loss(self, outputs, target):
        scale = max(1, int(self.args.stable_agg_scale))
        if outputs.shape[1] < scale:
            return outputs.new_tensor(0.0)
        valid_len = (outputs.shape[1] // scale) * scale
        outputs_agg = outputs[:, :valid_len, :].reshape(outputs.shape[0], valid_len // scale, scale, outputs.shape[2]).mean(dim=2)
        target_agg = target[:, :valid_len, :].reshape(target.shape[0], valid_len // scale, scale, target.shape[2]).mean(dim=2)
        return F.mse_loss(outputs_agg, target_agg)

    def _drift_loss(self, outputs, target):
        return F.mse_loss(self._extract_trend(outputs), self._extract_trend(target))

    def _measurement_loss(self, batch_x, batch_x_mark, dec_inp, batch_y_mark, outputs, target):
        mask_ratio = float(self.args.measurement_mask_ratio)
        if mask_ratio <= 0:
            return outputs.new_tensor(0.0)

        x_mask = (torch.rand_like(batch_x) > mask_ratio).float()
        masked_x = batch_x * x_mask
        if self.args.output_attention:
            masked_outputs = self.model(masked_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        else:
            masked_outputs = self.model(masked_x, batch_x_mark, dec_inp, batch_y_mark)

        f_dim = -1 if self.args.features == 'MS' else 0
        masked_outputs = masked_outputs[:, -self.args.pred_len:, f_dim:]
        omega = (torch.rand_like(target) < mask_ratio).float()
        recon = F.l1_loss(masked_outputs * omega, target * omega)

        error_signal = (masked_outputs - outputs.detach()).abs()
        logits = (error_signal - error_signal.mean(dim=1, keepdim=True))
        pattern = F.binary_cross_entropy_with_logits(logits, omega)
        return recon + pattern

    def _gradient_decorrelation(self, losses, outputs):
        valid = [loss for loss in losses if loss is not None]
        if len(valid) < 2:
            return outputs.new_tensor(0.0)

        grads = []
        for loss in valid:
            grad = torch.autograd.grad(loss, outputs, retain_graph=True, create_graph=True, allow_unused=True)[0]
            if grad is not None:
                grads.append(grad.reshape(grad.shape[0], -1).mean(dim=0))
        if len(grads) < 2:
            return outputs.new_tensor(0.0)

        reg = outputs.new_tensor(0.0)
        for g1, g2 in combinations(grads, 2):
            reg = reg + torch.abs(F.cosine_similarity(g1, g2, dim=0, eps=1e-8))
        return reg

    def _stage_lambdas(self):
        ls, ld, lm = self.args.lambda_s, self.args.lambda_d, self.args.lambda_m
        if not self.args.enable_btta:
            return 0.0, 0.0, 0.0
        if self.args.btta_stage == 'pretrain':
            return ls, 0.0, lm
        if self.args.btta_stage == 'bta':
            return ls, ld, lm
        if self.args.btta_stage == 'btta':
            return 0.0, 0.0, lm
        return ls, ld, lm

    def _compute_total_loss(self, batch_x, dec_inp, batch_y, batch_x_mark, batch_y_mark, criterion):
        if self.args.output_attention:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        else:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        target = batch_y[:, -self.args.pred_len:, f_dim:]
        task_loss = criterion(outputs, target)

        lambda_s, lambda_d, lambda_m = self._stage_lambdas()
        loss_s = self._stable_loss(outputs, target) if lambda_s > 0 else None
        loss_d = self._drift_loss(outputs, target) if lambda_d > 0 else None
        loss_m = self._measurement_loss(batch_x, batch_x_mark, dec_inp, batch_y_mark, outputs, target) if lambda_m > 0 else None

        total_loss = task_loss
        if loss_s is not None:
            total_loss = total_loss + lambda_s * loss_s
        if loss_d is not None:
            total_loss = total_loss + lambda_d * loss_d
        if loss_m is not None:
            total_loss = total_loss + lambda_m * loss_m

        if self.args.enable_btta and self.args.mu_grad > 0:
            grad_reg = self._gradient_decorrelation([loss_s, loss_d, loss_m], outputs)
            total_loss = total_loss + self.args.mu_grad * grad_reg

        return total_loss, outputs

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                
                
                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        c = nn.L1Loss()
        
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                # print('y start')
                batch_y = batch_y.float().to(self.device)
                
                # exit()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                #macs, params = profile(self.model, inputs=(batch_x, batch_x_mark, dec_inp, batch_y_mark))
                #macs, params = clever_format([macs, params], "%.3f")
                #print(macs, params)
                #exit()
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        
                        loss, _ = self._compute_total_loss(batch_x, dec_inp, batch_y, batch_x_mark, batch_y_mark, criterion)
                        train_loss.append(loss.item())
                else: 
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    if self.args.model == 'CARD':
                        self.ratio = np.array([max(1/np.sqrt(i+1),0.0) for i in range(self.args.pred_len)])
                        self.ratio = torch.tensor(self.ratio).unsqueeze(-1).to('cuda')
                        outputs = outputs *self.ratio
                        batch_y = batch_y *self.ratio
                        loss = c(outputs, batch_y)

                        use_h_loss = False
                        h_level_range = [4,8,16,24,48,96]
                        h_loss = None
                        if use_h_loss:                            
                            for h_level in h_level_range:
                                batch,length,channel = outputs.shape
                                # print(outputs.shape)
                                h_outputs = outputs.transpose(-1,-2).reshape(batch,channel,-1,h_level)
                                h_outputs = torch.mean(h_outputs,dim = -1,keepdims = True)
                                h_batch_y = batch_y.transpose(-1,-2).reshape(batch,channel,-1,h_level)
                                h_batch_y = torch.mean(h_batch_y,dim = -1,keepdims = True)
                                h_ratio = self.ratio[:h_outputs.shape[-2],:]
                                # print(h_outputs.shape,h_ratio.shape)
                                h_ouputs_agg = torch.mean(h_outputs,dim = 1,keepdims = True)
                                h_batch_y_agg = torch.mean(h_batch_y,dim = 1,keepdims = True)

                                h_outputs = h_outputs*h_ratio
                                h_batch_y = h_batch_y*h_ratio

                                h_ouputs_agg *= h_ratio
                                h_batch_y_agg *= h_ratio

                                if h_loss is None:
                                    h_loss  = c(h_outputs, h_batch_y)*np.sqrt(h_level) /2 +c(h_ouputs_agg, h_batch_y_agg)*np.sqrt(h_level) /2
                                else:
                                    h_loss = h_loss + c(h_outputs, h_batch_y)*np.sqrt(h_level) /2 +c(h_ouputs_agg, h_batch_y_agg)*np.sqrt(h_level) /2
                            # outputs = 0


                    else:
                        loss, _ = self._compute_total_loss(batch_x, dec_inp, batch_y, batch_x_mark, batch_y_mark, criterion)
                    
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    # visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        #np.save(folder_path + 'pred.npy', preds)
        #np.save(folder_path + 'true.npy', trues)

        return
