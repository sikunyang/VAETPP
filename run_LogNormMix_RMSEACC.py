import dpp
import numpy as np
import torch
import pickle
import argparse
import os
from math import sqrt
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
    np.random.seed(seed)
    torch.manual_seed(seed)
    # dataset_name = 'dataset_NYMVC_dim5' # run dpp.data.list_datasets() to see the list of available datasets
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

    dataset_name = 'dataset_NYMVC_dim5'
    dimension = 5  # number of dimensions
    num_prd = 5  # number of time intervals
    batch_size = 5

    # dataset_name = 'dataset_SUPERUSR_beta2_dim10'
    # dimension = 10  # number of dimensions
    # num_prd = 5  # number of time intervals
    # batch_size = 5

    # dataset_name = 'dataset_ASKUBUN_beta2_dim11'
    # dimension = 11  # number of dimensions
    # num_prd = 5  # number of time intervals
    # batch_size = 10

    # dataset_name = 'dataset_MATHOF_beta2_dim16'
    # dimension = 16  # number of dimensions
    # num_prd = 5  # number of time intervals
    # batch_size = 5

    print(dataset_name)
    # Model config
    context_size = 32               # Size of the RNN hidden vector
    mark_embedding_size = 32          # Size of the mark embedding (used as RNN input)
    num_mix_components = 32          # Number of components for a mixture model
    rnn_type = "GRU"                  # What RNN to use as an encoder {"RNN", "GRU", "LSTM"}

    # Training config
    # batch_size = 10       # Number of sequences in a batch
    regularization = 1e-5  # L2 regularization parameter
    learning_rate = 1e-3   # Learning rate for Adam optimizer
    max_epochs = 500      # For how many epochs to train
    display_step = 1     # Display training statistics after every display_step
    patience = 1000       # After how many consecutive epochs without improvement of val loss to stop training


    # Load the data

    dataset = dpp.data.load_dataset_mv(dataset_name)

    #############
    ## EUEMAIL
    # dimension = 10 # number of dimensions
    # num_prd = 30 # number of time intervals
    # batch_size = 5       # Number of sequences in a batch

    # #############
    # ## ASKUBUN
    # dimension = 8 # number of dimensions
    # num_prd = 7 # number of time intervals
    # batch_size = 5       # Number of sequences in a batch
    #
    # #############
    # ## SUPERUSR
    # dimension = 5 # number of dimensions
    # num_prd = 5 # number of time intervals
    # batch_size = 5

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
    dl_train = d_train.get_dataloader(batch_size=batch_size, shuffle=True)
    dl_val = d_val.get_dataloader(batch_size=batch_size, shuffle=False)
    dl_test = d_test.get_dataloader(batch_size=batch_size, shuffle=False)


    # Define the model
    # print('Building model...')
    mean_log_inter_time, std_log_inter_time = d_train.get_inter_time_statistics()

    model = dpp.models.LogNormMixUni(
        dimension = dimension,
        num_prd = num_prd,
        dimension_len = 199,
        num_marks=d_train.num_marks,
        mean_log_inter_time= mean_log_inter_time,
        std_log_inter_time=std_log_inter_time,
        context_size=context_size,
        mark_embedding_size=mark_embedding_size,
        rnn_type=rnn_type,
        num_mix_components=num_mix_components,
    )
    opt = torch.optim.Adam(model.parameters(), weight_decay=regularization, lr=learning_rate)


    # Traning
    # print('Starting training...')

    def aggregate_loss_over_dataloader(dl):
        total_loss = 0.0
        total_count = 0
        with torch.no_grad():
            for batch in dl:
                total_loss += -model.log_prob_v1(batch).sum()
                total_count += batch.size
        return total_loss / total_count

    def aggregate_loss_per_event_over_dataloader(dl):
        total_loss = 0.0
        total_count = 0
        total_loss = 0.0
        total_count = 0
        total_loglik = 0
        tot_pred_num = 0
        with torch.no_grad():
            for batch in dl:
        #         total_loss += -model.log_prob_v1(batch).sum()
        #         total_count += batch.mask.sum()
        # return total_loss / total_count
                dataloglik, se, loss_mark, pred_num = model.log_prob_v1(batch)# model.log_prob_with_dynamic_graph_v2(batch, mode)
                total_loss += -dataloglik.sum()
                # total_loss = 0
                total_loglik += se.sum()
                tot_pred_num += pred_num
                total_count += batch.mask.sum()
            return total_loss / total_count, sqrt(total_loglik / total_count), tot_pred_num / (total_count)

    impatient = 0
    best_loss = np.inf
    best_model = deepcopy(model.state_dict())
    training_val_losses = []

    for epoch in range(max_epochs):
        model.train()
        totse = 0
        tot_loss_mark = 0
        # total_count = 0
        for batch in dl_train:
            opt.zero_grad()
            # loss = -model.log_prob_v1(batch).mean()
            # # total_count += batch.mask.sum()
            # loss.backward()
            # opt.step()
            dataloglik, se, loss_mark, pred_num = model.log_prob_v1(batch)#model.log_prob_with_dynamic_graph_v2(batch, mode)
            # loss = -dataloglik
            # loss = loss.mean()
            ##
            loss = -dataloglik.sum() + se + loss_mark
            ####
            # loss = se
            ####
            loss.backward()
            opt.step()
            totse += se
            tot_loss_mark += loss_mark

        model.eval()
        with torch.no_grad():
            # loss_val = aggregate_loss_per_event_over_dataloader(dl_val)
            loss_val, negllk_val, acc_pred_mark = aggregate_loss_per_event_over_dataloader(dl_val)
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
            # print(f"Epoch {epoch:4d}: loss_train_last_batch = {loss.item():.1f}, loss_val = {loss_val:.1f}")
            print(f"Epoch {epoch:4d}: loss_train_last_batch = {loss.item():.1f}, loss_val = {loss_val:.1f},rmse_tr = {totse}, rmse_val = {negllk_val}, acc_marks = {acc_pred_mark}")

    # Evaluation
    model.load_state_dict(best_model)
    model.eval()

    # All training & testing sequences stacked into a single batch
    # with torch.no_grad():
    #     final_loss_train = aggregate_loss_per_event_over_dataloader(dl_train)
    #     final_loss_val = aggregate_loss_per_event_over_dataloader(dl_val)
    #     final_loss_test = aggregate_loss_per_event_over_dataloader(dl_test)
    with torch.no_grad():
        final_loss_train, final_se_train, final_acc_train = aggregate_loss_per_event_over_dataloader(dl_train)
        final_loss_val, final_se_val, final_acc_val = aggregate_loss_per_event_over_dataloader(dl_val)
        final_loss_test, final_se_test, final_acc_test = aggregate_loss_per_event_over_dataloader(dl_test)

    print(f'Negative log-likelihood:\n'
          f' - Train: {final_loss_train:.1f}\n'
          f' - Val:   {final_loss_val:.1f}\n'
          f' - Test:  {final_loss_test:.1f}')
    filepath = os.getcwd()
    name = '/results'
    if not os.path.exists(filepath+name):
        # name = '\TreeGraph'+str(J)+'Nodes'+str(rv_dim)+'Dims'+str(date.today())
        os.makedirs(filepath+name)
    # file1 = open(filepath + '/results/TestNLL-LogNormMix-' + dataset_name +'-'+ str(seed) + '.txt', "w")
    # file1.writelines(str(final_loss_test.numpy()))
    # file1.close()
    file1 = open(filepath + '/results/TestNLL-LogNormMix-' + dataset_name +'-'+ str(seed) + '.txt', "w")
    file1.writelines(str(final_loss_test.numpy()))
    file1.close()

    file1 = open(filepath + '/results/TestSE-LogNormMix-' + dataset_name +'-'+ str(seed) + '.txt', "w")
    file1.writelines(str(final_se_test))
    file1.close()

    file1 = open(filepath + '/results/TestACC-LogNormMix-' + dataset_name +'-'+ str(seed) + '.txt', "w")
    file1.writelines(str(final_acc_test.numpy()))
    file1.close()
