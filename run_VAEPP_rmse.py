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

# Config
for seed in range(5):
    # seed = seed + 3
    np.random.seed(seed)
    torch.manual_seed(seed)
    # dataset_name = 'dataset_NYMVC_dim5' # run dpp.data.list_datasets() to see the list of available datasets
    #
    # dataset_name = 'dataset_SUPERUSR_dim5'
    # dataset_name = 'dataset_MATHOF_dim7'
    # dataset_name = 'dataset_SOF_dim10'
    # dataset_name = 'dataset_EUEMAIL_dim10'
    # dataset_name = 'dataset_NYMVC_dim5'
    # dataset_name = 'dataset_ASKUBUN_dim8'
    # dataset_name = 'dataset_SUPERUSR_beta_dim5'
    # dataset_name = 'dataset_MATHOF_beta_dim5'
    # dataset_name = 'dataset_SOF_beta_dim10'
    # dataset_name = 'dataset_ASKUBUN_beta_dim8'
    dataset_name = 'dataset_SUPERUSR_beta2_dim10'
    dimension = 10  # number of dimensions
    num_prd = 5  # number of time intervals
    batch_size = 5

    # dataset_name = 'dataset_SUPERUSR_dim5'
    # dimension = 5  # number of dimensions
    # num_prd = 7  # number of time intervals
    # batch_size = 5

    # run dpp.data.list_datasets() to see the list of available datasets
    print(dataset_name)
    # Model config
    context_size = 32  # Size of the RNN hidden vector
    encoder_siz = 32 * 2
    mark_embedding_size = 32  # Size of the mark embedding (used as RNN input)
    num_mix_components = 32 * 4  # Number of components for a mixture model
    rnn_type = "GRU"  # What RNN to use as an encoder {"RNN", "GRU", "LSTM"}

    # Training config
    # dimension =    4
    ##############
    # EUEMAIL
    # dimension = 10  # number of dimensions
    # num_prd = 30  # number of time intervals
    # batch_size = 5  # Number of sequences in a batch

    #############
    ## ASKUBUN
    # dimension = 5  # number of dimensions
    # num_prd = 5  # number of time intervals
    # batch_size = 5  # Number of sequences in a batch

    # #############
    ## SUPERUSR
    # dimension = 5  # number of dimensions
    # num_prd = 7  # number of time intervals
    # batch_size = 5
    # #
    # # #############
    # # ## MATHFlow
    # dimension = 7  # number of dimensions
    # num_prd = 7  # number of time intervals
    # batch_size = 5
    #
    # #############
    # # SOFlow
    # dimension = 5  # number of dimensions
    # num_prd = 5  # number of time intervals
    # batch_size = 5

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
    max_epochs = 500*2    # For how many epochs to train
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
    dl_train = d_train.get_dataloader(batch_size=batch_size, shuffle=False)
    # dl_val = d_val.get_dataloader(batch_size=batch_size, shuffle=False)
    dl_val = d_train.get_dataloader(batch_size=batch_size, shuffle=False)
    dl_test = d_test.get_dataloader(batch_size=batch_size, shuffle=False)

    # Define the model
    print('Building model...')
    mean_log_inter_time, std_log_inter_time = d_train.get_inter_time_statistics()

    model = dpp.models.LogNormMixMv(
        dimension=dimension,
        num_prd=num_prd,
        dimension_len=dimension_len,
        num_marks=d_train.num_marks,
        mean_log_inter_time=mean_log_inter_time,
        std_log_inter_time=std_log_inter_time,
        context_size=context_size,
        encoder_siz=encoder_siz,
        mark_embedding_size=mark_embedding_size,
        rnn_type=rnn_type,
        num_mix_components=num_mix_components,
        num_edges=num_edges,
    )

    opt = torch.optim.Adam(model.parameters(), weight_decay=regularization, lr=learning_rate)

    # Traning
    print('Starting training...')


    def aggregate_loss_over_dataloader(dl, mode):
        total_loss = 0.0
        total_count = 0
        with torch.no_grad():
            for batch in dl:
                # total_loss += -model.log_prob_multivariate(batch).sum()
                total_loss += -model.log_prob_with_dynamic_graph_v1(batch, mode).sum()
                # total_loss += tt_loss.sum()
                total_count += batch.size
        return total_loss / total_count


    def aggregate_loss_per_event_over_dataloader(dl, mode):
        total_loss = 0.0
        total_count = 0
        with torch.no_grad():
            for batch in dl:
                # total_loss += -model.log_prob(batch).sum()
                # total_loss += -model.log_prob_with_dynamic_graph_v1(batch, mode).sum()
                dataloglik,se = model.log_prob_with_dynamic_graph_v1(batch, mode)
                total_loss = -dataloglik.sum()
                total_count += batch.mask.sum()
        return total_loss / total_count

    def aggregate_se_per_event_over_dataloader(dl, mode):
        total_se = 0.0
        total_count = 0
        with torch.no_grad():
            for batch in dl:
                # total_loss += -model.log_prob(batch).sum()
                # total_loss += -model.log_prob_with_dynamic_graph_v1(batch, mode).sum()
                dataloglik,se = model.log_prob_with_dynamic_graph_v1(batch, mode)
                total_se += se.item() #.sum()
                total_count += batch.size * batch.max_seq_len#batch.mask.sum()
        return total_se#/total_count

    impatient = 0
    best_loss = np.inf
    best_model = deepcopy(model.state_dict())
    training_val_losses = []

    for epoch in range(max_epochs):
        model.train()
        mode = 'Train'
        totse = 0
        total_count = 0
        for batch in dl_train:
            opt.zero_grad()
            # loss = -model.log_prob_multivariate(batch).mean()
            dataloglik, se = model.log_prob_with_dynamic_graph_v1(batch, mode)
            loss = -dataloglik
            loss = loss.mean()
            ####
            loss = se #/5
            ####
            loss.backward()
            opt.step()
            totse += se
            total_count += batch.mask.sum()
        model.eval()
        mode = 'Eval'
        # with torch.no_grad():
        #     loss_val = aggregate_loss_per_event_over_dataloader(dl_val, mode)
        #     training_val_losses.append(loss_val)
        with torch.no_grad():
            loss_val = aggregate_se_per_event_over_dataloader(dl_val, mode)
            training_val_losses.append(loss_val)


        if (best_loss - loss_val) < 1e-4:
            impatient += 1
            if loss_val < best_loss:
                best_loss = loss_val
                best_model = deepcopy(model.state_dict())
        else:
            best_loss = loss_val
            best_model = deepcopy(model.state_dict())
            impatient = 0

        if impatient >= patience:
            print(f'Breaking due to early stopping at epoch {epoch}')
            break

        if epoch % display_step == 0:
            print(f"Epoch {epoch:4d}: se_train = {totse}, se_val = {loss_val}")
            #print(f"Epoch {epoch:4d}")
            #print(f"Totse {totse.item():.1f}")
            #print(f"Valse {loss_val.item():.1f}")
    # Evaluation
    model.load_state_dict(best_model)
    model.eval()
    mode = 'Eval'

    # All training & testing sequences stacked into a single batch
    with torch.no_grad():
        final_loss_train = aggregate_se_per_event_over_dataloader(dl_train, mode)
        final_loss_val = aggregate_se_per_event_over_dataloader(dl_val, mode)
        final_loss_test = aggregate_se_per_event_over_dataloader(dl_test, mode)

    print(f'SE:\n'
          f' - Train: {final_loss_train}\n'
          f' - Val:   {final_loss_val}\n'
          f' - Test:  {final_loss_test}')

    filepath = os.getcwd()
    name = '/results'
    if not os.path.exists(filepath+name):
        # name = '\TreeGraph'+str(J)+'Nodes'+str(rv_dim)+'Dims'+str(date.today())
        os.makedirs(filepath+name)
    file1 = open(filepath + '/results/TestSE-VAETPP-' + dataset_name +'-'+ str(seed) + '.txt', "w")
    file1.writelines(str(final_loss_test.numpy()))
    file1.close()