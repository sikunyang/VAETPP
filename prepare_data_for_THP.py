import dpp
import numpy as np
import torch
import pickle
import argparse
import os
from copy import deepcopy
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
def get_inter_times_uni(seq: dict, ind):
    """Get inter-event times from a sequence."""
    temp1 = seq["t_start"]
    temp2 = seq["arr_times"][ind]
    return np.ediff1d(np.concatenate([[seq["t_start"]], seq["arr_times"][ind]]))  # , [seq["t_end"]]

mypath = '/Users/yangsikun/TokyoCityFlow/'
mypath = mypath + 'THPdata/NYMVC/'
dataset_name = 'dataset_MATHOF_dim7'
# dataset_name = 'dataset_SUPERUSR_dim5'

# dataset_name = 'dataset_SOF_dim10'
# dataset_name = 'dataset_EUEMAIL_dim10'
dataset_name = 'dataset_NYMVC_dim5'
# dataset_name = 'dataset_ASKUBUN_dim8'

# Config
for seed in range(5):
    dirname = mypath + 'fold' + str(seed + 1) + '/'
    # seed = seed + 3
    np.random.seed(seed)
    torch.manual_seed(seed)
    # dataset_name = 'dataset_NYMVC_dim5' # run dpp.data.list_datasets() to see the list of available datasets
    #


    # run dpp.data.list_datasets() to see the list of available datasets
    print(dataset_name)
    # Model config
    context_size = 32  # Size of the RNN hidden vector
    encoder_siz = 32 * 2
    mark_embedding_size = 32  # Size of the mark embedding (used as RNN input)
    num_mix_components = 32  # Number of components for a mixture model
    rnn_type = "GRU"  # What RNN to use as an encoder {"RNN", "GRU", "LSTM"}

    # Training config
    # dimension = 4
    ##############
    ## EUEMAIL
    # dimension = 10  # number of dimensions
    # num_prd = 30  # number of time intervals
    # batch_size = 5  # Number of sequences in a batch

    #############
    ## ASKUBUN
    # dimension = 8  # number of dimensions
    # num_prd = 7  # number of time intervals
    # batch_size = 5  # Number of sequences in a batch

    # #############
    ## SUPERUSR
    # dimension = 5  # number of dimensions
    # num_prd = 7  # number of time intervals
    # batch_size = 5
    #
    # #############
    # ## MATHFlow
    # dimension = 7  # number of dimensions
    # num_prd = 7  # number of time intervals
    # batch_size = 5
    #
    # #############
    # SOFlow
    dimension = 10  # number of dimensions
    num_prd = 7  # number of time intervals
    batch_size = 5

    # #############
    # ## synthetic
    # dimension = 4
    # num_prd = 6
    # batch_size = 5
    #
    # ##############
    # # ## NYMVC
    # dimension = 5  # number of dimensions
    # num_prd = 5  # number of time intervals
    # batch_size = 5

    ##############
    regularization = 1e-5  # L2 regularization parameter
    learning_rate = 1e-3  # Learning rate for Adam optimizer
    max_epochs = 500  # For how many epochs to train
    display_step = 1  # Display training statistics after every display_step
    patience = 500  # After how many consecutive epochs without improvement of val loss to stop training
    num_edges = 2

    # Load the data

    dataset = dpp.data.load_dataset_mv(dataset_name)
    ### NYMVC

    nseq = len(dataset.sequences)
    max_seq_len = 0
    for iseq in range(nseq):
        temp = len(dataset.sequences[iseq].inter_times)
        # temp = len(dataset.sequences[iseq].marks)+1
        if temp > max_seq_len:
            max_seq_len = temp
    dimension_len = max_seq_len
    dimension_len = int(dimension_len)

    d_train, d_val, d_test = dataset.train_val_test_split(seed=seed)

    ###### valid
    tempdata = []
    totnumseq = len(d_train.sequences)
    # totnumtrainseq = int(totnumseq * 0.6)
    # totnumtestseq = int(totnumseq * 0.2)
    for s in range(totnumseq):
        seq = []
        seq_idx = s
        LastTime = 0
        seq_len = len(d_train.sequences[seq_idx]['arr_times'])
        ###
        inter_times = np.zeros((1, len(d_train.sequences[seq_idx]["arr_times"])))
        for d in range(len(np.unique(d_train.sequences[seq_idx]['marks']))):
            ind = np.where(d_train.sequences[seq_idx]['marks'] == d)
            temp = get_inter_times_uni(d_train.sequences[seq_idx], ind)
            temp[np.where(temp == 0)] = 1
            inter_times[0, ind] = np.log(temp)  # \
        # to_end = [seq["t_end"] - seq["arrival_times"][-1]]
        # inter_times[0, len(seq["arrival_times"])] = to_end[0]
        ###

        for ie in range(seq_len):
            eventname = str(ie).zfill(3)
            # timediff = dataset[seq_idx]['arrival_times'][ie] - LastTime
            # if timediff == 0:
            #     timediff = 1
            globals()[eventname] = {'time_since_start': np.log(d_train.sequences[seq_idx]['arr_times'][ie]),
                                    'time_since_last_event': inter_times[0][ie],
                                    'type_event': d_train.sequences[seq_idx]['marks'][ie]}
            # globals()[eventname] = {'time_since_start': inter_times[0][ie],
            #                         'time_since_last_event': inter_times[0][ie],
            #                         'type_event': dataset[seq_idx]['marks'][ie]}

            seq.append(globals()[eventname])
            del globals()[eventname]
            # LastTime = dataset[seq_idx]['arrival_times'][ie]
        seqname = str(ie).zfill(5)
        globals()[seqname] = seq
        tempdata.append(globals()[seqname])
        del globals()[seqname]

    # train_data = [tempdata[idx] for idx in train_idxs]  # tempdata[train_idxs]
    data = {'test1': [], 'args': None, 'dim_process': 7, 'dev': [], 'train': tempdata, 'test': []}
    file = open(dirname + 'train.pkl', 'wb')
    pickle.dump(data, file)
    file.close()

    ###### valid
    tempdata = []
    totnumseq = len(d_val.sequences)
    # totnumtrainseq = int(totnumseq * 0.6)
    # totnumtestseq = int(totnumseq * 0.2)
    for s in range(totnumseq):
        seq = []
        seq_idx = s
        LastTime = 0
        seq_len = len(d_val.sequences[seq_idx]['arr_times'])
        ###
        inter_times = np.zeros((1, len(d_val.sequences[seq_idx]["arr_times"])))
        for d in range(len(np.unique(d_val.sequences[seq_idx]['marks']))):
            ind = np.where(d_val.sequences[seq_idx]['marks'] == d)
            temp = get_inter_times_uni(d_val.sequences[seq_idx], ind)
            temp[np.where(temp == 0)] = 1
            inter_times[0, ind] = np.log(temp)  # \
        # to_end = [seq["t_end"] - seq["arrival_times"][-1]]
        # inter_times[0, len(seq["arrival_times"])] = to_end[0]
        ###

        for ie in range(seq_len):
            eventname = str(ie).zfill(3)
            # timediff = dataset[seq_idx]['arrival_times'][ie] - LastTime
            # if timediff == 0:
            #     timediff = 1
            globals()[eventname] = {'time_since_start':  np.log(d_val.sequences[seq_idx]['arr_times'][ie]), 'time_since_last_event': inter_times[0][ie] ,'type_event': d_val.sequences[seq_idx]['marks'][ie]}
            # globals()[eventname] = {'time_since_start': inter_times[0][ie],
            #                         'time_since_last_event': inter_times[0][ie],
            #                         'type_event': dataset[seq_idx]['marks'][ie]}

            seq.append(globals()[eventname])
            del globals()[eventname]
            # LastTime = dataset[seq_idx]['arrival_times'][ie]
        seqname = str(ie).zfill(5)
        globals()[seqname] = seq
        tempdata.append(globals()[seqname])
        del globals()[seqname]

    # train_data = [tempdata[idx] for idx in train_idxs]  # tempdata[train_idxs]
    data = {'test1': [], 'args': None, 'dim_process': 7, 'dev': tempdata, 'train': [], 'test': []}
    file = open(dirname + 'dev.pkl', 'wb')
    pickle.dump(data, file)
    file.close()

    ###### test
    tempdata = []
    totnumseq = len(d_test.sequences)
    # totnumtrainseq = int(totnumseq * 0.6)
    # totnumtestseq = int(totnumseq * 0.2)
    for s in range(totnumseq):
        seq = []
        seq_idx = s
        LastTime = 0
        seq_len = len(d_test.sequences[seq_idx]['arr_times'])
        ###
        inter_times = np.zeros((1, len(d_test.sequences[seq_idx]["arr_times"])))
        for d in range(len(np.unique(d_test.sequences[seq_idx]['marks']))):
            ind = np.where(d_test.sequences[seq_idx]['marks'] == d)
            temp = get_inter_times_uni(d_test.sequences[seq_idx], ind)
            temp[np.where(temp == 0)] = 1
            inter_times[0, ind] = np.log(temp)  # \
        # to_end = [seq["t_end"] - seq["arrival_times"][-1]]
        # inter_times[0, len(seq["arrival_times"])] = to_end[0]
        ###

        for ie in range(seq_len):
            eventname = str(ie).zfill(3)
            # timediff = dataset[seq_idx]['arrival_times'][ie] - LastTime
            # if timediff == 0:
            #     timediff = 1
            globals()[eventname] = {'time_since_start':  np.log(d_test.sequences[seq_idx]['arr_times'][ie]), 'time_since_last_event': inter_times[0][ie] ,'type_event': d_test.sequences[seq_idx]['marks'][ie]}
            # globals()[eventname] = {'time_since_start': inter_times[0][ie],
            #                         'time_since_last_event': inter_times[0][ie],
            #                         'type_event': dataset[seq_idx]['marks'][ie]}

            seq.append(globals()[eventname])
            del globals()[eventname]
            # LastTime = dataset[seq_idx]['arrival_times'][ie]
        seqname = str(ie).zfill(5)
        globals()[seqname] = seq
        tempdata.append(globals()[seqname])
        del globals()[seqname]

    # train_data = [tempdata[idx] for idx in train_idxs]  # tempdata[train_idxs]
    data = {'test1': [], 'args': None, 'dim_process': 7, 'dev': [], 'train': [], 'test': tempdata}
    file = open(dirname + 'test.pkl', 'wb')
    pickle.dump(data, file)
    file.close()

    print('d')