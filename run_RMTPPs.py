import dpp_prev
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
for seed in range(3):
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset_name = 'dataset_NYMVC_dim5' # run dpp.data.list_datasets() to see the list of available datasets
    # dataset_name = 'dataset_SUPERUSR_dim5'
    # dataset_name = 'dataset_MATHOF_dim7'

    # dataset_name = 'dataset_SUPERUSR_beta_dim5'
    # dataset_name = 'dataset_MATHOF_beta_dim5'
    # dataset_name = 'dataset_SOF_beta_dim10'
    # dataset_name = 'dataset_ASKUBUN_beta_dim8'
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
    max_epochs = 200      # For how many epochs to train
    display_step = 1     # Display training statistics after every display_step
    patience = 1000       # After how many consecutive epochs without improvement of val loss to stop training


    # Load the data

    dataset = dpp.data.load_dataset_mv(dataset_name)



    # #############
    # ## EUEMAIL
    # dimension = 10 # number of dimensions
    # num_prd = 30 # number of time intervals
    # batch_size = 5       # Number of sequences in a batch
    #
    # #############
    # ## ASKUBUN
    # dimension = 8 # number of dimensions
    # num_prd = 7 # number of time intervals
    # batch_size = 5       # Number of sequences in a batch

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
    dl_train = d_train.get_dataloader(batch_size=batch_size, shuffle=True)
    dl_val = d_val.get_dataloader(batch_size=batch_size, shuffle=False)
    dl_test = d_test.get_dataloader(batch_size=batch_size, shuffle=False)

    mean_out_train, std_out_train = d_train.get_inter_time_statistics()

    #### RMTPPs
    decoder_name = 'RMTPP'
    # other: ['RMTPP', 'FullyNeuralNet', 'Exponential', 'SOSPolynomial', 'DeepSigmoidalFlow']
    n_components = 64  # Number of components for a mixture model
    hypernet_hidden_sizes = []  # Number of units in MLP generating parameters ([] -- affine layer, [64] -- one layer, etc.)

    ## Flow params
    # Polynomial
    max_degree = 3  # Maximum degree value for Sum-of-squares polynomial flow (SOS)
    n_terms = 4  # Number of terms for SOS flow
    # DSF / FullyNN
    n_layers = 2  # Number of layers for Deep Sigmoidal Flow (DSF) / Fully Neural Network flow (Omi et al., 2019)
    layer_size = 64  # Number of mixture components / units in a layer for DSF and FullyNN

    ## Training config
    regularization = 1e-5  # L2 regularization parameter
    learning_rate = 1e-3  # Learning rate for Adam optimizer
    max_epochs = 1000  # For how many epochs to train
    display_step = 1  # Display training statistics after every display_step
    patience = 50  # After how many consecutive epochs without improvement of val loss to stop training

    # Set the parameters for affine normalization layer depending on the decoder (see Appendix D.3 in the paper)
    if decoder_name in ['RMTPP', 'FullyNeuralNet', 'Exponential']:
        #_, std_out_train = d_train.get_mean_std_out()
        mean_out_train = 0.0
    else:
        mean_out_train, std_out_train = d_train.get_log_mean_std_out()

    ### Model setup
    print('Building model...')

    # General model config
    ## General model config
    use_history = True  # Whether to use RNN to encode history
    history_size = 64  # Size of the RNN hidden vector
    rnn_type = 'RNN'  # Which RNN cell to use (other: ['GRU', 'LSTM'])
    use_embedding = False  # Whether to use sequence embedding (should use with 'each_sequence' split)
    embedding_size = 32  # Size of the sequence embedding vector
    # IMPORTANT: when using split = 'whole_sequences', the model will only learn embeddings
    # for the training sequences, and not for validation / test
    trainable_affine = False  # Train the final affine layer

    general_config = dpp_prev.model.ModelConfig(
        use_history=use_history,
        history_size=history_size,
        rnn_type=rnn_type,
        use_embedding=use_embedding,
        embedding_size=embedding_size,
        num_embeddings=len(dataset),
    )

    # Decoder specific config
    decoder = getattr(dpp_prev.decoders, decoder_name)(general_config,
                                                  n_components=n_components,
                                                  hypernet_hidden_sizes=hypernet_hidden_sizes,
                                                  max_degree=max_degree,
                                                  n_terms=n_terms,
                                                  n_layers=n_layers,
                                                  layer_size=layer_size,
                                                  shift_init=mean_out_train,
                                                  scale_init=std_out_train,
                                                  trainable_affine=trainable_affine)

    # Define model
    model = dpp_prev.model.Model(general_config, decoder)
    model.use_history(general_config.use_history)
    model.use_embedding(general_config.use_embedding)

    # Define optimizer
    opt = torch.optim.Adam(model.parameters(), weight_decay=regularization, lr=learning_rate)

    ### Traning
    print('Starting training...')


    # Function that calculates the loss for the entire dataloader
    def get_total_loss(loader):
        loader_log_prob, loader_lengths = [], []
        for input in loader:
            loader_log_prob.append(model.log_prob(input).detach())
            loader_lengths.append(input.length.detach())
        return -model.aggregate(loader_log_prob, loader_lengths)


    impatient = 0
    best_loss = np.inf
    best_model = deepcopy(model.state_dict())
    training_val_losses = []

    for epoch in range(max_epochs):
        model.train()
        for input in dl_train:
            opt.zero_grad()
            log_prob = model.log_prob(input)
            loss = -model.aggregate(log_prob, input.length)
            loss.backward()
            opt.step()

        model.eval()
        loss_val = get_total_loss(dl_val)
        training_val_losses.append(loss_val.item())

        if (best_loss - loss_val) < 1e-4:
            impatient += 1
            if loss_val < best_loss:
                best_loss = loss_val.item()
                best_model = deepcopy(model.state_dict())
        else:
            best_loss = loss_val.item()
            best_model = deepcopy(model.state_dict())
            impatient = 0

        if impatient >= patience:
            print(f'Breaking due to early stopping at epoch {epoch}')
            break

        if (epoch + 1) % display_step == 0:
            print(f"Epoch {epoch + 1:4d}, loss_train_last_batch = {loss:.4f}, loss_val = {loss_val:.4f}")

    ### Evaluation

    model.load_state_dict(best_model)
    model.eval()

    pdf_loss_train = get_total_loss(dl_train)
    pdf_loss_val = get_total_loss(dl_val)
    pdf_loss_test = get_total_loss(dl_test)

    print(f'Time NLL\n'
          f' - Train: {pdf_loss_train:.4f}\n'
          f' - Val:   {pdf_loss_val.item():.4f}\n'
          f' - Test:  {pdf_loss_test.item():.4f}')

    ####
    # # Define the model
    # # print('Building model...')
    # mean_log_inter_time, std_log_inter_time = d_train.get_inter_time_statistics()
    #
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
    #
    #
    # # Traning
    # # print('Starting training...')
    #
    # def aggregate_loss_over_dataloader(dl):
    #     total_loss = 0.0
    #     total_count = 0
    #     with torch.no_grad():
    #         for batch in dl:
    #             total_loss += -model.log_prob(batch).sum()
    #             total_count += batch.size
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
    # best_model = deepcopy(model.state_dict())
    # training_val_losses = []
    #
    # for epoch in range(max_epochs):
    #     model.train()
    #     for batch in dl_train:
    #         opt.zero_grad()
    #         loss = -model.log_prob(batch).mean()
    #         loss.backward()
    #         opt.step()
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
    filepath = os.getcwd()
    name = '/results'
    if not os.path.exists(filepath+name):
        # name = '\TreeGraph'+str(J)+'Nodes'+str(rv_dim)+'Dims'+str(date.today())
        os.makedirs(filepath+name)
    file1 = open(filepath + '/results/TestNLL-RMTPPs-' + dataset_name +'-'+ str(seed) + '.txt', "w")
    file1.writelines(str(pdf_loss_test.numpy()))
    file1.close()
