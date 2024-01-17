import dpp
import numpy as np
import torch
import pickle
import argparse
import os
from copy import deepcopy
from argparse import ArgumentParser
from module import GTPP
from utils import read_timeseries,generate_sequence, plt_lmbda
from torch.utils.data import DataLoader

# torch.set_default_tensor_type(torch.cuda.FloatTensor)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# parser = argparse.ArgumentParser(description='build single compare instance')
# # parser.add_argument('-method', action='append', default=[])
# parser.add_argument('-dataset', action='append', default=[])
# result = parser.parse_args()
# if result.method[0] == 'MOT-IOT-lowrank':
#     parser.add_argument('-rank', action='append', default=[])
# result.method = ['CNP']
# result.dataset = ['SF']

# Config
for seed in range(3):
    np.random.seed(seed)
    torch.manual_seed(seed)
    # dataset_name = 'dataset_NYMVC_dim5' # run dpp.data.list_datasets() to see the list of available datasets
    dataset_name = 'dataset_SUPERUSR_dim5'
    print(dataset_name)
    # Model config
    context_size = 32               # Size of the RNN hidden vector
    mark_embedding_size = 32          # Size of the mark embedding (used as RNN input)
    num_mix_components = 32          # Number of components for a mixture model
    rnn_type = "GRU"                  # What RNN to use as an encoder {"RNN", "GRU", "LSTM"}

    # Training config
    batch_size = 30       # Number of sequences in a batch
    regularization = 1e-5  # L2 regularization parameter
    learning_rate = 1e-3   # Learning rate for Adam optimizer
    max_epochs = 500      # For how many epochs to train
    display_step = 1     # Display training statistics after every display_step
    patience = 1000       # After how many consecutive epochs without improvement of val loss to stop training



    dataset = dpp.data.load_dataset_mv(dataset_name)

    #############
    ## EUEMAIL
    dimension = 10 # number of dimensions
    num_prd = 30 # number of time intervals
    batch_size = 5       # Number of sequences in a batch

    #############
    ## ASKUBUN
    dimension = 8 # number of dimensions
    num_prd = 7 # number of time intervals
    batch_size = 5       # Number of sequences in a batch

    #############
    ## SUPERUSR
    dimension = 5 # number of dimensions
    num_prd = 5 # number of time intervals
    batch_size = 5

    #############
    # # MATHFlow
    # dimension = 7 # number of dimensions
    # num_prd = 7 # number of time intervals
    # batch_size = 5

    # #############
    # ## SOFlow
    # dimension = 10 # number of dimensions
    # num_prd = 7 # number of time intervals
    # batch_size = 5
    #
    # #############
    # ## synthetic
    # dimension = 4
    # num_prd = 6
    # batch_size = 5
    #
    # ##############
    # ## NYMVC
    # dimension = 5 # number of dimensions
    # num_prd = 5 # number of time intervals
    # batch_size = 5

    d_train, d_val, d_test = dataset.train_val_test_split(seed=seed)
    #####
    ### train
    tempdata = []
    for s in range(len(d_train)):
        # print(s)
        # s = s+1
        # samp = df[df['CRASH DATE'] == '07/'+str(s).zfill(2)+'/2021']

        # date = datetime.datetime.strptime('07/'+str(s).zfill(2)+'/2021'+' 00:00:00', "%m/%d/%Y %H:%M:%S")
        # date = dates[s]
        # samp = df[df['CRASH DATE'] == date]
        # date2unixtime = datetime.datetime.strptime(date + ' 00:00:00', "%m/%d/%Y %H:%M:%S")
        # start = datetime.datetime.timestamp(date2unixtime)
        # TSS = samp['unixtime'] - start
        arrtime = np.array(d_train[s].arr_times)
        # inter_time = np.array(d_train[s].arr_times)# arrtime[1:] - arrtime[:-1]
        # TSLE = np.insert(inter_time, 0, [0])
        # TYPES = np.array(samp['RE'])
        TYPES = np.array(d_train[s].marks)
        inter_time = np.array(d_train[s].inter_times)
        seq = []
        for ie in range(len(TYPES)):
            eventname = str(ie).zfill(3)
            # globals()[eventname] = {'time_since_start': arrtime[ie], #'time_since_last_event': TSLE[ie],
            #                         'type_event': TYPES[ie]}
            globals()[eventname] = (arrtime[ie],TYPES[ie],inter_time[ie])
            seq.append(globals()[eventname])
            del globals()[eventname]
        seqname = str(ie).zfill(5)
        globals()[seqname] = seq
        tempdata.append(globals()[seqname])
        del globals()[seqname]
    # print('done_traindata')
    train_data = tempdata
    #####
    tempdata = []
    for s in range(len(d_val)):
        arrtime = np.array(d_val[s].arr_times)
        TYPES = np.array(d_val[s].marks)
        inter_time = np.array(d_val[s].inter_times)
        seq = []
        for ie in range(len(TYPES)):
            eventname = str(ie).zfill(3)
            globals()[eventname] = (arrtime[ie], TYPES[ie],inter_time[ie])
            seq.append(globals()[eventname])
            del globals()[eventname]
        seqname = str(ie).zfill(5)
        globals()[seqname] = seq
        tempdata.append(globals()[seqname])
        del globals()[seqname]
    # print('done_valdata')
    val_data = tempdata
    #####
    tempdata = []
    for s in range(len(d_test)):
        arrtime = np.array(d_test[s].arr_times)
        TYPES = np.array(d_test[s].marks)
        inter_time = np.array(d_test[s].inter_times)
        seq = []
        for ie in range(len(TYPES)):
            eventname = str(ie).zfill(3)
            globals()[eventname] = (arrtime[ie], TYPES[ie],inter_time[ie])
            seq.append(globals()[eventname])
            del globals()[eventname]
        seqname = str(ie).zfill(5)
        globals()[seqname] = seq
        tempdata.append(globals()[seqname])
        del globals()[seqname]
    # print('done_testdata')
    test_data = tempdata

    # dl_train = d_train.get_dataloader(batch_size=batch_size, shuffle=True)
    # dl_val = d_val.get_dataloader(batch_size=batch_size, shuffle=False)
    # dl_test = d_test.get_dataloader(batch_size=batch_size, shuffle=False)

    ### FullyNN
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, default='exponential_hawkes')
    parser.add_argument("--model", type=str, default='GTPP')
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--emb_dim", type=int, default=10)
    parser.add_argument("--hid_dim", type=int, default=64)
    parser.add_argument("--mlp_layer", type=int, default=2)
    parser.add_argument("--mlp_dim", type=int, default=64)
    parser.add_argument("--event_class", type=int, default=dimension)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=float, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--prt_evry", type=int, default=5)
    parser.add_argument("--early_stop", type=bool, default=False)
    ## Alpha ??
    parser.add_argument("--alpha", type=float, default=0.05)

    parser.add_argument("--importance_weight", action="store_true")
    parser.add_argument("--log_mode", type=bool, default=False)
    config = parser.parse_args()
    model = GTPP(config)
    # Load the data
    train_timeseq, train_eventseq = generate_sequence(train_data, config.seq_len, log_mode=config.log_mode)
    train_loader = DataLoader(torch.utils.data.TensorDataset(train_timeseq, train_eventseq), shuffle=True, batch_size=config.batch_size)
    val_timeseq, val_eventseq = generate_sequence(val_data, config.seq_len, log_mode=config.log_mode)
    val_loader = DataLoader(torch.utils.data.TensorDataset(val_timeseq, val_eventseq), shuffle=False,
                            batch_size=len(val_data))
    test_timeseq, test_eventseq = generate_sequence(test_data, config.seq_len, log_mode=config.log_mode)
    test_loader = DataLoader(torch.utils.data.TensorDataset(test_timeseq, test_eventseq), shuffle=False,
                            batch_size=len(test_data))

    best_loss = 1e3
    patients = 0
    tol = 30
    # Define the model
    # print('Building model...')
    # mean_log_inter_time, std_log_inter_time = d_train.get_inter_time_statistics()

    # model = dpp.models.LogNormMixUni(
    #     dimension = dimension,
    #     num_prd = num_prd,
    #     dimension_len = 199,
    #     num_marks=d_train.num_marks,
    #     mean_log_inter_time=mean_log_inter_time,
    #     std_log_inter_time=std_log_inter_time,
    #     context_size=context_size,
    #     mark_embedding_size=mark_embedding_size,
    #     rnn_type=rnn_type,
    #     num_mix_components=num_mix_components,
    # )
    # opt = torch.optim.Adam(model.parameters(), weight_decay=regularization, lr=learning_rate)


    # Traning
    # print('Starting training...')
    for epoch in range(config.epochs):

        model.train()

        loss1 = loss2 = loss3 = 0

        for batch in train_loader:
            loss, log_lmbda, int_lmbda, lmbda = model.train_batch(batch)

            loss1 += loss
            loss2 += log_lmbda
            loss3 += int_lmbda


        model.eval()

        for batch in val_loader:
            val_loss, val_log_lmbda, val_int_lmbda, _ = model(batch)

        if best_loss > val_loss:
            best_loss = val_loss.item()
        else:
            patients += 1
            # if patients >= tol:
            #     print("Early Stop")
            #     print("epoch", epoch)
            #     plt_lmbda(train_data[0], model=model, seq_len=config.seq_len, log_mode=config.log_mode)
            #     break

        # if epoch % config.prt_evry == 0:
        #     print("Epochs:{}".format(epoch))
        #     print("Training Negative Log Likelihood:{}   Log Lambda:{}:   Integral Lambda:{}".format(loss1 #/train_timeseq.size(0)
        #                                                                                              , -loss2 / train_timeseq.size(0), loss3 / train_timeseq.size(0)))
        #     print("Validation Negative Log Likelihood:{}   Log Lambda:{}:   Integral Lambda:{}".format(val_loss# / val_timeseq.size(0)
        #                                                                                             , -val_log_lmbda / val_timeseq.size(0),
        #                                                                                     val_int_lmbda/val_timeseq.size(0)))
        #     plt_lmbda(train_data[0], model=model, seq_len=config.seq_len, log_mode=config.log_mode, epoch = epoch)
        #     plt_lmbda(test_data[0], model=model, seq_len=config.seq_len, log_mode=config.log_mode)
    model.eval()

    loss1 = loss2 = loss3 = 0
    for batch in train_loader:
        train_loss, train_log_lmbda, train_int_lmbda, _ = model(batch)
        loss1 += train_loss
        loss2 += train_log_lmbda
        loss3 += train_int_lmbda

    print("=====Training Negative Log Likelihood:{}".format(loss1 / train_timeseq.size(0)))

    loss1 = loss2 = loss3 = 0
    for batch in val_loader:
        val_loss, val_log_lmbda, val_int_lmbda, _ = model(batch)
        loss1 += val_loss
        loss2 += val_log_lmbda
        loss3 += val_int_lmbda

    print("=====Validation Negative Log Likelihood:{}".format(loss1 / val_timeseq.size(0)))

    loss1 = loss2 = loss3 = 0
    for batch in test_loader:
        test_loss, test_log_lmbda, test_int_lmbda, _ = model(batch)
        loss1 += test_loss
        loss2 += test_log_lmbda
        loss3 += test_int_lmbda

    print("=====Testing Negative Log Likelihood:{}".format(loss1/test_timeseq.size(0)))


    print("end")

    # def aggregate_loss_over_dataloader(dl):
    #     total_loss = 0.0
    #     total_count = 0
    #     with torch.no_grad():
    #         for batch in dl:
    #             # total_loss += -model.log_prob(batch).sum()
    #             # total_count += batch.size
    #             ### FullyNN
    #             loss, log_lmbda, int_lmbda, lmbda = model.train_batch(batch)
    #             loss1 += loss
    #             loss2 += log_lmbda
    #             loss3 += int_lmbda
    #             ###
    #     return total_loss / total_count
    #
    # def aggregate_loss_per_event_over_dataloader(dl):
    #     total_loss = 0.0
    #     total_count = 0
    #     with torch.no_grad():
    #         for batch in dl:
    #             total_loss += -model.log_prob(batch).sum()
    #             total_count += batch.mask.sum()
    #     return total_loss / total_count
    #
    # impatient = 0
    # best_loss = np.inf
    # # best_model = deepcopy(model.state_dict())
    # training_val_losses = []
    #
    # nseq = len(dataset.sequences)
    # max_seq_len = 0
    # for iseq in range(nseq):
    #     temp = len(dataset.sequences[iseq].inter_times)
    #     # temp = len(dataset.sequences[iseq].marks)+1
    #     if temp > max_seq_len:
    #         max_seq_len = temp
    # dimension_len = max_seq_len
    # dimension_len = int(dimension_len)
    #

    #
    # for epoch in range(max_epochs):
    #     model.train()
    #     ### FullyNN
    #     loss1 = loss2 = loss3 = 0
    #     for batch in dl_train:
    #         # opt.zero_grad()
    #         # loss = -model.log_prob(batch).mean()
    #         ### FullyNN
    #         loss, log_lmbda, int_lmbda, lmbda = model.train_batch(batch)
    #         loss1 += loss
    #         loss2 += log_lmbda
    #         loss3 += int_lmbda
    #         ###
    #         # loss.backward()
    #         # opt.step()
    #
    #     model.eval()
    #     with torch.no_grad():
    #         loss_val = aggregate_loss_over_dataloader(dl_val)
    #         training_val_losses.append(loss_val)
    #
    #     if (best_loss - loss_val) < 1e-4:
    #         impatient += 1
    #         if loss_val < best_loss:
    #             best_loss = loss_val
    #             best_model = deepcopy(model.state_dict())
    #     else:
    #         best_loss = loss_val
    #         best_model = deepcopy(model.state_dict())
    #         impatient = 0
    #
    #     if impatient >= patience:
    #         print(f'Breaking due to early stopping at epoch {epoch}')
    #         break
    #
    #     if epoch % display_step == 0:
    #         print(f"Epoch {epoch:4d}: loss_train_last_batch = {loss.item():.1f}, loss_val = {loss_val:.1f}")
    #
    #
    # # Evaluation
    # model.load_state_dict(best_model)
    # model.eval()
    #
    # # All training & testing sequences stacked into a single batch
    # with torch.no_grad():
    #     final_loss_train = aggregate_loss_per_event_over_dataloader(dl_train)
    #     final_loss_val = aggregate_loss_per_event_over_dataloader(dl_val)
    #     final_loss_test = aggregate_loss_per_event_over_dataloader(dl_test)
    #
    # print(f'Negative log-likelihood:\n'
    #       f' - Train: {final_loss_train:.1f}\n'
    #       f' - Val:   {final_loss_val:.1f}\n'
    #       f' - Test:  {final_loss_test:.1f}')
    # filepath = os.getcwd()
    # name = '/results'
    # if not os.path.exists(filepath+name):
    #     # name = '\TreeGraph'+str(J)+'Nodes'+str(rv_dim)+'Dims'+str(date.today())
    #     os.makedirs(filepath+name)
    # file1 = open(filepath + '/results/TestNLL-LogNormMix-' + dataset_name +'-'+ str(seed) + '.txt', "w")
    # file1.writelines(str(final_loss_test.numpy()))
    # file1.close()
