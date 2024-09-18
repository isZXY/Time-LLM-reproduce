import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from data_provider.data_factory import data_provider
import random
from torch.optim import lr_scheduler

import numpy as np
import argparse
from models import TimeLLM
import os
from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali,vali_no_accelerator, load_content
import time
from tqdm import tqdm


# 参数解析
parser = argparse.ArgumentParser(description='Time-LLM')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# basic config
parser.add_argument('--task_name', type=str, required=True, default='classification',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model_comment', type=str, required=True, default='none', help='prefix when saving test results')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, DLinear]')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; '
                         'M:multivariate predict multivariate, S: univariate predict univariate, '
                         'MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--loader', type=str, default='modal', help='dataset type')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length') # 输入序列的长度，即模型每次输入的时间步数（通常是历史数据的长度）。
parser.add_argument('--label_len', type=int, default=48, help='start token length') # 起始标记长度，通常指的是模型在生成预测时所用的最后一段已知序列的长度。
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length') # 预测序列的长度，即模型需要预测的未来时间步数。
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4') # 季节性模式

# classification task
# parser.add_argument('--num_classes',type=int,default=7,help='classification category class number')

# model define
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # 输入维度，即输入特征的数量，应该是N
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size') # 解码器所需的输入特征数量
parser.add_argument('--c_out', type=int, default=7, help='output size') # 模型的输出维度，即输出特征的数量
parser.add_argument('--d_model', type=int, default=16, help='dimension of model') # patch模型的隐藏层维度
parser.add_argument('--n_heads', type=int, default=8, help='num of heads') # 多头注意力机制中头的数量
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers') # 编码器的层数
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers') # 解码器的层数
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn') # 前馈神经网络的维度
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average') # 移动平均窗口的大小
parser.add_argument('--factor', type=int, default=1, help='attn factor') # 注意力机制中的缩放因子
parser.add_argument('--dropout', type=float, default=0.1, help='dropout') # dropout 的比例
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]') # 时间特征的编码方式
parser.add_argument('--activation', type=str, default='gelu', help='activation') # 激活函数
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder') # 是否在编码器中输出注意力权重
parser.add_argument('--patch_len', type=int, default=16, help='patch length') # patch）长度
parser.add_argument('--stride', type=int, default=8, help='stride') # 步幅
parser.add_argument('--prompt_domain', type=int, default=0, help='')
parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model') # LLAMA, GPT2, BERT
parser.add_argument('--llm_dim', type=int, default='4096', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768


# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--percent', type=int, default=100)

args = parser.parse_args()

for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.des, ii)
    # 数据准备
    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 模型加载
    model = TimeLLM.Model(args).float()
    model.to(device)
    path = os.path.join(args.checkpoints,
                        setting + '-' + args.model_comment)  # unique checkpoint saving path
    args.content = load_content(args) # 加载prompt
    if not os.path.exists(path):
        os.makedirs(path)

    time_now = time.time()

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(patience=args.patience)

    trained_parameters = []
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)
    # 定义损失函数和优化器
    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate) # 优化器


    # 学习率调度器
    if args.lradj == 'COS':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
    else:
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=args.pct_start,
                                            epochs=args.train_epochs,
                                            max_lr=args.learning_rate)
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    mae_metric = nn.L1Loss()



    # 训练循环
    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []

        model.train()
        epoch_time = time.time()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)): # 原始sequence,预测sequence，各自对应的时间戳 
            iter_count += 1
            model_optim.zero_grad()

            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(device) # 后面全0，用于预测，相当于mask了？ (选取了 batch_y 中最后 pred_len 长度的序列)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device) # 前面就是要使用的
            # encoder - decoder
            ## 改: 结构
            ## encoder(with or without amp)
            if args.output_attention:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())

            if (i + 1) % 100 == 0:
                print(
                    "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()


            loss.backward()
            model_optim.step()

            # if args.lradj == 'TST':
            #     adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
            #     scheduler.step()

        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)
        vali_loss, vali_mae_loss = vali_no_accelerator(args, device, model, vali_data, vali_loader, criterion, mae_metric)
        test_loss, test_mae_loss = vali_no_accelerator(args, device, model, test_data, test_loader, criterion, mae_metric)
        print(
            "Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f} MAE Loss: {4:.7f}".format(
                epoch + 1, train_loss, vali_loss, test_loss, test_mae_loss))

        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        if args.lradj != 'TST':
            if args.lradj == 'COS':
                scheduler.step()
                print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                if epoch == 0:
                    args.learning_rate = model_optim.param_groups[0]['lr']
                    print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                adjust_learning_rate(model_optim, scheduler, epoch + 1, args, printout=True)

        else:
            print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))


    path = './checkpoints'  # unique checkpoint saving path
    del_files(path)  # delete checkpoint files
    print('success delete checkpoints')