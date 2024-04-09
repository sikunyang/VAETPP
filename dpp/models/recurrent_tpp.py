import dpp
import timeit
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical

from sklearn.metrics import mean_squared_error
from math import sqrt

from dpp.data.batch import Batch
from dpp.utils import diff
import numpy as np
# from .model_utils import encode_onehot, RefNRIMLP

class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        super(LabelSmoothingLoss, self).__init__()

        self.eps = label_smoothing
        self.num_classes = tgt_vocab_size
        self.ignore_index = ignore_index

    def forward(self, output, target):
        """
        output (FloatTensor): (batch_size) x n_classes
        target (LongTensor): batch_size
        """

        non_pad_mask = target.ne(self.ignore_index).float()

        target[target.eq(self.ignore_index)] = 0
        one_hot = F.one_hot(target, num_classes=self.num_classes).float()
        one_hot = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / self.num_classes

        log_prb = F.log_softmax(output, dim=-1)
        loss = -(one_hot * log_prb).sum(dim=-1)
        loss = loss * non_pad_mask
        return loss

class RefNRIMLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0., no_bn=False):
        super(RefNRIMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_in, n_hid),
            nn.ELU(inplace=True),
            nn.Dropout(do_prob),
            nn.Linear(n_hid, n_out),
            nn.ELU(inplace=True)
        )
        if no_bn:
            self.bn = None
        else:
            self.bn = nn.BatchNorm1d(n_out)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        orig_shape = inputs.shape
        x = inputs.view(-1, inputs.size(-1))
        x = self.bn(x)
        return x.view(orig_shape)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = self.model(inputs)
        if self.bn is not None:
            return self.batch_norm(x)
        else:
            return x

class Predictor(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, dim, num_types):
        super().__init__()

        self.linear = nn.Linear(dim, num_types, bias=False)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, data):
        out = self.linear(data)
        # out = out * non_pad_mask
        return out

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def sample_gumbel(shape, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3
    Sample from Gumbel(0, 1)
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = torch.rand(shape).float()
    return - torch.log(eps - torch.log(U + eps))

def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3
    Draw a sample from the Gumbel-Softmax distribution
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    y = logits + gumbel_noise
    return F.softmax(y / tau, dim=-1)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3
    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      tau: non-negative scalar temperature
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes
    Constraints:
    - this implementation only works on batch_size x num_features tensor for now
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros(*shape)
        if y_soft.is_cuda:
            y_hard = y_hard.cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = y_hard - y_soft.data + y_soft
    else:
        y = y_soft
    return y

class RecurrentTPP(nn.Module):
    """
    RNN-based TPP model for marked and unmarked event sequences.

    The marks are assumed to be conditionally independent of the inter-event times.

    Args:
        num_marks: Number of marks (i.e. classes / event types)
        mean_log_inter_time: Average log-inter-event-time, see dpp.data.dataset.get_inter_time_statistics
        std_log_inter_time: Std of log-inter-event-times, see dpp.data.dataset.get_inter_time_statistics
        context_size: Size of the context embedding (history embedding)
        mark_embedding_size: Size of the mark embedding (used as RNN input)
        rnn_type: Which RNN to use, possible choices {"RNN", "GRU", "LSTM"}

    """
    def __init__(
        self,
        dimension: int,
        num_prd: int,
        dimension_len: int,
        num_marks: int,
        mean_log_inter_time: float = 0.0,
        std_log_inter_time: float = 1.0,
        context_size: int = 32,
        encoder_hidsiz: int = 64,
        mark_embedding_size: int = 32,
        rnn_type: str = "GRU",
        num_edges: int = 2,
    ):
        super().__init__()
        self.dimension = dimension
        self.num_prd = num_prd
        self.dimension_len = dimension_len
        self.num_marks = num_marks
        self.mean_log_inter_time = mean_log_inter_time
        self.std_log_inter_time = std_log_inter_time
        self.context_size = context_size
        self.mark_embedding_size = mark_embedding_size
        if self.num_marks > 1:
            self.num_features = 1 + self.mark_embedding_size
            self.mark_embedding = nn.Embedding(self.num_marks, self.mark_embedding_size)
            self.mark_linear = nn.Linear(self.context_size, self.num_marks)
        else:
            self.num_features = 1
        self.rnn_type = rnn_type
        self.rnn = getattr(nn, rnn_type)(input_size=self.num_features, hidden_size=self.context_size, batch_first=True)

        self.context_init = nn.Parameter(torch.zeros(context_size))  # initial state of the RNN
        fwd_type = rnn_type
        self.fwd = getattr(nn, fwd_type)(input_size=self.num_features, hidden_size=self.context_size, batch_first=True)
        self.num_edges = num_edges


        ##########
        # encoder
        dropout = 0.5
        no_bn = False
        self.factor = True
        self.encoder_hidsiz = encoder_hidsiz
        inp_siz = 1
        self.mlp1_v1 = RefNRIMLP(inp_siz, encoder_hidsiz, encoder_hidsiz, dropout, no_bn=no_bn)

        self.mlp1 = RefNRIMLP(dimension_len, encoder_hidsiz, encoder_hidsiz, dropout, no_bn=no_bn)
        self.mlp2 = RefNRIMLP(encoder_hidsiz * 2, encoder_hidsiz, encoder_hidsiz, dropout, no_bn=no_bn)
        self.mlp3 = RefNRIMLP(encoder_hidsiz, encoder_hidsiz, encoder_hidsiz, dropout, no_bn=no_bn)
        if self.factor:
            self.mlp4 = RefNRIMLP(encoder_hidsiz * 3, encoder_hidsiz, encoder_hidsiz, dropout, no_bn=no_bn)
            print("Using factor graph MLP encoder.")
        else:
            self.mlp4 = RefNRIMLP(encoder_hidsiz * 2, encoder_hidsiz, encoder_hidsiz, dropout, no_bn=no_bn)
            print("Using MLP encoder.")

        tmp_hidden_size = encoder_hidsiz# self.context_size# 64#256
        num_layers = 1#3
        num_edges = 3 #self.dimension * (self.dimension-1)
        if num_layers == 1:
            self.fc_out = nn.Linear(tmp_hidden_size, num_edges)
        else:
            layers = [nn.Linear(encoder_hidsiz, tmp_hidden_size), nn.ELU(inplace=True)]
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(tmp_hidden_size, tmp_hidden_size))
                layers.append(nn.ELU(inplace=True))
            layers.append(nn.Linear(tmp_hidden_size, num_edges))
            self.fc_out = nn.Sequential(*layers)
        self.gumbel_temp = 0.5#0.75

        ##########

        self.dropout_prob = 0.5
        self.msg_out_shape = self.context_size
        self.skip_first_edge_type = True
        num_vertex = self.dimension
        self.num_nodes = num_vertex
        edges = np.ones(num_vertex) - np.eye(num_vertex)
        # edges[0,2]=0
        # edges[0,3]=0
        # edges[1,2]=0
        # edges[1,3]=0
        # edges[2,0]=0
        # edges[2,1]=0
        # edges[3,0]=0
        # edges[3,1]=0
        edge_types = num_edges
        # batch_size = 60
        # rel_type = torch.zeros((num_vertex*(num_vertex-1), edge_types))
        # rel_type[0, 1] = 1
        # rel_type[3, 1] = 1
        # rel_type[8, 1] = 1
        # rel_type[11, 1] = 1
        #
        # # rel_type[0, 0] = 1
        # # rel_type[3, 0] = 1
        # # rel_type[8, 0] = 1
        # # rel_type[11, 0] = 1
        #
        # rel_type[1, 0] = 1
        # rel_type[2, 0] = 1
        # rel_type[4, 0] = 1
        # rel_type[5, 0] = 1
        # rel_type[6, 0] = 1
        # rel_type[7, 0] = 1
        # rel_type[9, 0] = 1
        # rel_type[10, 0] = 1

        # self.rel_type = rel_type#.unsqueeze(0).expand(batch_size,rel_type.size(0),rel_type.size(1))
        # tmp = np.where(edges)
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]
        # self.edge2node_mat = torch.FloatTensor(self.encode_onehot(self.recv_edges))
        self.node2edge_mat = nn.Parameter(torch.FloatTensor(encode_onehot(self.recv_edges).transpose()),requires_grad=False)
        self.edge2node_mat = torch.FloatTensor(encode_onehot(self.recv_edges))

        # if self.gpu:
        #     self.edge2node_mat = self.edge2node_mat.cuda(non_blocking=True)

        n_hid = self.context_size#150#64 * 2
        self.hidden_r = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_i = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_h = nn.Linear(n_hid, n_hid, bias=False)

        input_size = 1
        self.input_r = nn.Linear(input_size, n_hid, bias=True)
        self.input_i = nn.Linear(input_size, n_hid, bias=True)
        self.input_n = nn.Linear(input_size, n_hid, bias=True)

        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * n_hid, n_hid) for _ in range(edge_types)]
        )
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(n_hid, n_hid) for _ in range(edge_types)]
        )

        self.out_fc1 = nn.Linear(n_hid, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_hid)

        self.forward_rnn = nn.GRU(encoder_hidsiz, encoder_hidsiz, batch_first=True)
        self.reverse_rnn = nn.GRU(encoder_hidsiz, encoder_hidsiz, batch_first=True)
        out_hidden_size = 2 * encoder_hidsiz
        self.encoder_fc_out = nn.Linear(out_hidden_size, self.num_edges)
        self.prior_fc_out = nn.Linear(encoder_hidsiz, self.num_edges)
        self.encoder_fc_out_v1 = nn.Linear(self.dimension_len*encoder_hidsiz * 2, self.num_edges * self.num_prd)

        self.time_predictor = nn.Linear(n_hid * (self.num_nodes + 1), 1) # vaetpp
        self.mark_predictor = nn.Linear(n_hid * (self.num_nodes + 1), num_vertex)  # vaetpp

        self.time_predictor_v2 = nn.Linear(n_hid, 1)  # vaetpp
        self.mark_predictor_v2 = nn.Linear(n_hid, num_vertex)

        self.loss_mark_func = LabelSmoothingLoss(0.1, num_vertex, ignore_index=-1)
        # self.time_predictor = nn.Linear(n_hid, 1) # rnn
        # self.time_pred1 = nn.Linear(n_hid * self.num_nodes, 1)
        self.time_pred1 = nn.Linear(n_hid, 1)
        self.time_pred2 = nn.Linear(1, 1)
        self.time_pred3 = nn.Linear(1, 1)

        ######
        n_periods = self.num_prd
        self.compressor = nn.Linear(dimension_len, n_periods, bias=None)

        ###
        # self.num_edges_types = 2
        prior = np.zeros(self.num_edges)
        prior.fill((1 - 0.5) / (self.num_edges - 1))
        prior[0] = 0.5
        log_prior = torch.FloatTensor(np.log(prior))
        log_prior = torch.unsqueeze(log_prior, 0)
        log_prior = torch.unsqueeze(log_prior, 0)
        # if params['gpu']:
        #     log_prior = log_prior.cuda(non_blocking=True)
        self.log_prior = log_prior

    def encode_onehot(self, labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                        enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                 dtype=np.int32)
        return labels_onehot

    def kl_categorical(self, preds, eps=1e-16):
        kl_div = preds*(torch.log(preds+eps) - self.log_prior)
        if True:
            return kl_div.sum(-1).view(preds.size(0), -1).mean(dim=1)
        elif self.normalize_kl_per_var:
            return kl_div.sum() / (self.num_vars * preds.size(0))
        else:
            return kl_div.view(preds.size(0), -1).sum(dim=1)

    def kl_categorical_learned(self, preds, prior_logits):
        log_prior = nn.LogSoftmax(dim=-1)(prior_logits)
        kl_div = preds * (torch.log(preds + 1e-16) - log_prior)
        # if True:#self.normalize_kl:
        return kl_div.sum(-1).view(preds.size(0), -1).mean(dim=1)
        # elif self.normalize_kl_per_var:
        #     return kl_div.sum() / (self.num_vars * preds.size(0))
        # else:
        #     return kl_div.view(preds.size(0), -1).sum(dim=1)

    def get_features(self, batch: dpp.data.Batch) -> torch.Tensor:
        """
        Convert each event in a sequence into a feature vector.

        Args:
            batch: Batch of sequences in padded format (see dpp.data.batch).

        Returns:
            features: Feature vector corresponding to each event,
                shape (batch_size, seq_len, num_features)

        """
        # features = (batch.inter_times + 1e-8).unsqueeze(-1)  # (batch_size, seq_len, 1)
        features = (batch.inter_times.clamp(1e-10)).unsqueeze(-1)  # (batch_size, seq_len, 1)
        # features = (features - self.mean_log_inter_time) / self.std_log_inter_time
        if self.num_marks > 1:
            mark_emb = self.mark_embedding(batch.marks)  # (batch_size, seq_len, mark_embedding_size)
            features = torch.cat([features, mark_emb], dim=-1)
        return features  # (batch_size, seq_len, num_features)

    def get_context(self, features: torch.Tensor, remove_last: bool = True) -> torch.Tensor:
        """
        Get the context (history) embedding from the sequence of events.

        Args:
            features: Feature vector corresponding to each event,
                shape (batch_size, seq_len, num_features)
            remove_last: Whether to remove the context embedding for the last event.

        Returns:
            context: Context vector used to condition the distribution of each event,
                shape (batch_size, seq_len, context_size) if remove_last == False
                shape (batch_size, seq_len + 1, context_size) if remove_last == True

        """
        context = self.rnn(features)[0]
        batch_size, seq_len, context_size = context.shape
        context_init = self.context_init[None, None, :].expand(batch_size, 1, -1)  # (batch_size, 1, context_size)
        # Shift the context by vectors by 1: context embedding after event i is used to predict event i + 1
        if remove_last:
            context = context[:, :-1, :]
        context = torch.cat([context_init, context], dim=1)
        return context

    def get_inter_time_dist(self, context: torch.Tensor) -> torch.distributions.Distribution:
        """
        Get the distribution over inter-event times given the context.

        Args:
            context: Context vector used to condition the distribution of each event,
                shape (batch_size, seq_len, context_size)

        Returns:
            dist: Distribution over inter-event times, has batch_shape (batch_size, seq_len)

        """
        raise NotImplementedError()

    def log_prob(self, batch: dpp.data.Batch) -> torch.Tensor:
        """Compute log-likelihood for a batch of sequences.

        Args:
            batch:

        Returns:
            log_p: shape (batch_size,)

        """
        features = self.get_features(batch)
        context = self.get_context(features)

        ###
        # rnn_input = features
        # time_steps = rnn_input.size(1)
        # hid_size = self.context_size#150#128
        # hidden_state = torch.zeros(rnn_input.size(0), hid_size)
        # hidden_seq = []
        # for istep in range(time_steps):
        #     inp_r = self.input_r(rnn_input[:, istep, :])  # .view(inputs.size(0), self.num_vars, -1)
        #     inp_i = self.input_i(rnn_input[:, istep, :])  # .view(inputs.size(0), self.num_vars, -1)
        #     inp_n = self.input_n(rnn_input[:, istep, :])  # .view(inputs.size(0), self.num_vars, -1)
        #     # tmp = self.hidden_r(hidden_state)
        #     r = torch.sigmoid(inp_r + self.hidden_r(hidden_state))
        #     i = torch.sigmoid(inp_i + self.hidden_i(hidden_state))
        #     n = torch.tanh(inp_n + r * self.hidden_h(hidden_state))
        #     hidden_state = ((1 - i) * n + i * hidden_state)
        #     hidden_seq.append(hidden_state)
        # context = torch.stack(hidden_seq, dim=1)
        batch_size, seq_len, context_size = context.shape
        context_init = self.context_init[None, None, :].expand(batch_size, 1, -1)  # (batch_size, 1, context_size)
        # Shift the context by vectors by 1: context embedding after event i is used to predict event i + 1
        if True:
            context = context[:, :-1, :]
        context = torch.cat([context_init, context], dim=1)

        ###
        inter_time_dist = self.get_inter_time_dist(context)
        inter_times = batch.inter_times.clamp(1e-10)
        log_p = inter_time_dist.log_prob(inter_times)  # (batch_size, seq_len)

        # Survival probability of the last interval (from t_N to t_end).
        # You can comment this section of the code out if you don't want to implement the log_survival_function
        # for the distribution that you are using. This will make the likelihood computation slightly inaccurate,
        # but the difference shouldn't be significant if you are working with long sequences.
        last_event_idx = batch.mask.sum(-1, keepdim=True).long()  # (batch_size, 1)
        log_surv_all = inter_time_dist.log_survival_function(inter_times)  # (batch_size, seq_len)
        log_surv_last = torch.gather(log_surv_all, dim=-1, index=last_event_idx).squeeze(-1)  # (batch_size,)

        if self.num_marks > 1:
            mark_logits = torch.log_softmax(self.mark_linear(context), dim=-1)  # (batch_size, seq_len, num_marks)
            mark_dist = Categorical(logits=mark_logits)
            log_p += mark_dist.log_prob(batch.marks)  # (batch_size, seq_len)
        log_p *= batch.mask  # (batch_size, seq_len)
        return log_p.sum(-1) + log_surv_last  # (batch_size,)

    def log_prob_v1(self, batch: dpp.data.Batch) -> torch.Tensor:
        """Compute log-likelihood for a batch of sequences.

        Args:
            batch:

        Returns:
            log_p: shape (batch_size,)

        """
        features = self.get_features(batch)
        context = self.get_context(features)
        # context0 =  context
        context0 = self.fwd(features)[0]
        ###
        # rnn_input = features
        # time_steps = rnn_input.size(1)
        # hid_size = self.context_size#150#128
        # hidden_state = torch.zeros(rnn_input.size(0), hid_size)
        # hidden_seq = []
        # for istep in range(time_steps):
        #     inp_r = self.input_r(rnn_input[:, istep, :])  # .view(inputs.size(0), self.num_vars, -1)
        #     inp_i = self.input_i(rnn_input[:, istep, :])  # .view(inputs.size(0), self.num_vars, -1)
        #     inp_n = self.input_n(rnn_input[:, istep, :])  # .view(inputs.size(0), self.num_vars, -1)
        #     # tmp = self.hidden_r(hidden_state)
        #     r = torch.sigmoid(inp_r + self.hidden_r(hidden_state))
        #     i = torch.sigmoid(inp_i + self.hidden_i(hidden_state))
        #     n = torch.tanh(inp_n + r * self.hidden_h(hidden_state))
        #     hidden_state = ((1 - i) * n + i * hidden_state)
        #     hidden_seq.append(hidden_state)
        # context = torch.stack(hidden_seq, dim=1)
        batch_size, seq_len, context_size = context.shape
        context_init = self.context_init[None, None, :].expand(batch_size, 1, -1)  # (batch_size, 1, context_size)
        # Shift the context by vectors by 1: context embedding after event i is used to predict event i + 1
        if True:
            context = context[:, :-1, :]
        context = torch.cat([context_init, context], dim=1)

        ###
        inter_time_dist = self.get_inter_time_dist(context)
        inter_times = batch.inter_times.clamp(1e-10)
        log_p = inter_time_dist.log_prob(inter_times)  # (batch_size, seq_len)

        # Survival probability of the last interval (from t_N to t_end).
        # You can comment this section of the code out if you don't want to implement the log_survival_function
        # for the distribution that you are using. This will make the likelihood computation slightly inaccurate,
        # but the difference shouldn't be significant if you are working with long sequences.
        last_event_idx = batch.mask.sum(-1, keepdim=True).long()  # (batch_size, 1)
        log_surv_all = inter_time_dist.log_survival_function(inter_times)  # (batch_size, seq_len)
        log_surv_last = torch.gather(log_surv_all, dim=-1, index=last_event_idx).squeeze(-1)  # (batch_size,)

        if self.num_marks > 1:
            mark_logits = torch.log_softmax(self.mark_linear(context), dim=-1)  # (batch_size, seq_len, num_marks)
            mark_dist = Categorical(logits=mark_logits)
            log_p += mark_dist.log_prob(batch.marks)  # (batch_size, seq_len)
        log_p *= batch.mask  # (batch_size, seq_len)

        # return log_p.sum(-1) + log_surv_last

        pred_mark = self.mark_predictor_v2(context0)
        true_mark = batch.marks[:, 1:] - 1
        pred_mark = pred_mark[:, :-1, :]

        prediction = torch.max(pred_mark, dim=-1)[1]
        correct_num = torch.sum(prediction == true_mark)

        loss_mark = self.loss_mark_func(pred_mark.float(), true_mark)
        loss_mark = torch.sum(loss_mark)
        ######################################################
        pred_time = self.time_predictor_v2(context0)
        pred_time = pred_time.squeeze_(-1)
        ##########
        for iii in range(len(last_event_idx)):
            inter_times[iii, last_event_idx] = 0
        inter_times = inter_times.clamp(1e-10)

        # true = inter_times[:, 1:]
        # pred_time = pred_time[:, :-1]
        # true = inter_times[:, 1:]
        # prediction = pred_time[:, :-1]

        # # event time gap prediction
        # diff = prediction - true
        # rmse = torch.sum(diff * diff)
        # pred_time = inter_time_dist.mean


        #  inter_times = batch.inter_times.clamp(1e-10)

        # rmse = sqrt(mean_squared_error(pred_time,inter_times))
        # print('RMSE:',rmse)


        diff = pred_time - inter_times
        tmp = diff * diff
        rmse = torch.sum(tmp)
        # rmse = torch.tensor(0)


        # ####$$$$
        # ######################################
        # features = self.get_features(batch)
        # context0 = self.fwd(features)[0]
        # ####
        # batch_size, seq_len, context_size = context0.shape
        # context = torch.zeros((batch_size, seq_len, self.num_nodes, context_size))
        # context = context.view(context.size(0), context.size(1), -1)
        #
        # # inter_time_dist = self.get_inter_time_dist(context)
        # #
        # inter_times = batch.inter_times.clamp(1e-10)
        # #
        # # log_p = inter_time_dist.log_prob(inter_times)  # (batch_size, seq_len)
        # #
        # # last_event_idx = batch.mask.sum(-1, keepdim=True).long()  # (batch_size, 1)
        # # log_surv_all = inter_time_dist.log_survival_function(inter_times)  # (batch_size, seq_len)
        # # log_surv_last = torch.gather(log_surv_all, dim=-1, index=last_event_idx).squeeze(-1)  # (batch_size,)
        #
        # # log_p *= batch.mask  # (batch_size, seq_len)
        #
        # #########
        # context = torch.cat([context, context0], dim=-1)
        #
        # # pred_mark = self.mark_predictor(context)
        # # true_mark = batch.marks[:, 1:] - 1
        # # pred_mark = pred_mark[:, :-1, :]
        # #
        # # prediction = torch.max(pred_mark, dim=-1)[1]
        # # correct_num = torch.sum(prediction == true_mark)
        # #
        # # loss_mark = self.loss_mark_func(pred_mark.float(), true_mark)
        # # loss_mark = torch.sum(loss_mark)
        # # ######################################################
        # pred_time = self.time_predictor(context)
        # pred_time = pred_time.squeeze_(-1)
        # ##########
        # for iii in range(len(last_event_idx)):
        #     inter_times[iii, last_event_idx] = 0
        # inter_times = inter_times.clamp(1e-10)
        #
        # # true = inter_times[:, 1:]
        # # prediction = pred_time[:, :-1]
        #
        # diff = pred_time - inter_times
        # tmp = diff * diff
        # rmse = torch.sum(tmp)
        # ######$$$

        return log_p.sum(-1) + log_surv_last, rmse, loss_mark, correct_num

    def log_prob_multivariate(self, batch: dpp.data.Batch) -> torch.Tensor:
        """Compute log-likelihood for a batch of sequences.

        Args:
            batch:

        Returns:
            log_p: shape (batch_size,)

        """

        # context = self.get_context(features)
        ######################################
        ## decoder
        ######################################
        features = self.get_features(batch)
        ndim = self.dimension
        numbatch = batch.inter_times.size(0)
        expanded_features = torch.zeros(batch.inter_times.size(0), ndim, batch.inter_times.size(1))
        for ibatch in range(numbatch):
            for i in range(ndim):
                ind = torch.where(batch.marks[ibatch,:,]==i)#[idx for idx, element in enumerate(mixseq_marks) if mixseq_marks==i]
                expanded_features[ibatch,i,ind[0]] = batch.inter_times[ibatch,ind[0]]
            # uni_EF = expanded_features[ibatch,:,:]
        ###
        rnn_input = expanded_features
        time_steps = rnn_input.size(2)
        hid_size = 64
        hidden_state = torch.zeros(rnn_input.size(0), hid_size)
        hidden_seq = []
        for istep in range(time_steps):
            inp_r = self.input_r(rnn_input[:, :, istep])  # .view(inputs.size(0), self.num_vars, -1)
            inp_i = self.input_i(rnn_input[:, :, istep])  # .view(inputs.size(0), self.num_vars, -1)
            inp_n = self.input_n(rnn_input[:, :, istep])  # .view(inputs.size(0), self.num_vars, -1)
            tmp = self.hidden_r(hidden_state)
            r = torch.sigmoid(inp_r + self.hidden_r(hidden_state))# agg_msgs
            i = torch.sigmoid(inp_i + self.hidden_i(hidden_state))#
            n = torch.tanh(inp_n + r * self.hidden_h(hidden_state))#
            hidden_state = ((1 - i) * n + i * hidden_state)
            hidden_seq.append(hidden_state)
        context = torch.stack(hidden_seq, dim=1)
        batch_size, seq_len, context_size = context.shape
        context_init = self.context_init[None, None, :].expand(batch_size, 1, -1)  # (batch_size, 1, context_size)
        # Shift the context by vectors by 1: context embedding after event i is used to predict event i + 1
        if True:
            context = context[:, :-1, :]
        context = torch.cat([context_init, context], dim=1)

        ###
        inter_time_dist = self.get_inter_time_dist(context)
        inter_times = batch.inter_times.clamp(1e-10)
        log_p = inter_time_dist.log_prob(inter_times)  # (batch_size, seq_len)

        # Survival probability of the last interval (from t_N to t_end).
        # You can comment this section of the code out if you don't want to implement the log_survival_function
        # for the distribution that you are using. This will make the likelihood computation slightly inaccurate,
        # but the difference shouldn't be significant if you are working with long sequences.
        last_event_idx = batch.mask.sum(-1, keepdim=True).long()  # (batch_size, 1)
        log_surv_all = inter_time_dist.log_survival_function(inter_times)  # (batch_size, seq_len)
        log_surv_last = torch.gather(log_surv_all, dim=-1, index=last_event_idx).squeeze(-1)  # (batch_size,)

        if self.num_marks > 1:
            mark_logits = torch.log_softmax(self.mark_linear(context), dim=-1)  # (batch_size, seq_len, num_marks)
            mark_dist = Categorical(logits=mark_logits)
            log_p += mark_dist.log_prob(batch.marks)  # (batch_size, seq_len)
        log_p *= batch.mask  # (batch_size, seq_len)
        return log_p.sum(-1) + log_surv_last  # (batch_size,)

    def log_prob_with_graph(self, batch: dpp.data.Batch) -> torch.Tensor:
        """Compute log-likelihood for a batch of sequences.

        Args:
            batch:

        Returns:
            log_p: shape (batch_size,)

        """

        # context = self.get_context(features)
        features = self.get_features(batch)

        ndim = self.dimension
        numbatch = batch.inter_times.size(0)
        expanded_features = torch.zeros(batch.inter_times.size(0), ndim, batch.inter_times.size(1))
        for ibatch in range(numbatch):
            for i in range(ndim):
                ind = torch.where(batch.marks[ibatch,
                                  :, ] == i)  # [idx for idx, element in enumerate(mixseq_marks) if mixseq_marks==i]
                expanded_features[ibatch, i, ind[0]] = batch.inter_times[ibatch, ind[0]]
            # uni_EF = expanded_features[ibatch,:,:]
        ###
        rnn_input = expanded_features
        time_steps = rnn_input.size(2)
        hid_size = self.context_size  # 150#64 * 2
        hidden_state = torch.zeros(rnn_input.size(0), rnn_input.size(1), hid_size)
        hidden_seq = []
        for istep in range(time_steps):
            #####
            # node2edge
            receivers = hidden_state[:, self.recv_edges, :]
            senders = hidden_state[:, self.send_edges, :]

            # pre_msg: [batch, num_edges, 2*msg_out]
            pre_msg = torch.cat([receivers, senders], dim=-1)

            # if inputs.is_cuda:
            #     all_msgs = torch.cuda.FloatTensor(pre_msg.size(0), pre_msg.size(1),
            #                                       self.msg_out_shape).fill_(0.)
            # else:
            all_msgs = torch.zeros(pre_msg.size(0), pre_msg.size(1),
                                   self.msg_out_shape)

            if self.skip_first_edge_type:
                start_idx = 1
                norm = float(len(self.msg_fc2)) - 1
            else:
                start_idx = 0
                norm = float(len(self.msg_fc2))

            # Run separate MLP for every edge type
            # NOTE: to exclude one edge type, simply offset range by 1
            # tmp = len(self.msg_fc2)
            rel_type = self.rel_type.unsqueeze(0).expand(pre_msg.size(0), self.rel_type.size(0), self.rel_type.size(1))
            for i in range(start_idx, len(self.msg_fc2)):
                msg = torch.tanh(self.msg_fc1[i](pre_msg))
                msg = F.dropout(msg, p=self.dropout_prob)
                msg = torch.tanh(self.msg_fc2[i](msg))
                msg = msg * rel_type[:, :, i:i + 1]
                all_msgs += msg / norm

            # This step sums all of the messages per node
            agg_msgs = all_msgs.transpose(-2, -1).matmul(self.edge2node_mat).transpose(-2, -1)
            agg_msgs = agg_msgs.contiguous() / (self.num_nodes - 1)  # Average
            #####
            ins = rnn_input[:, :, istep].unsqueeze(-1)
            inp_r = self.input_r(ins)  # .view(inputs.size(0), self.num_vars, -1)
            inp_i = self.input_i(ins)  # .view(inputs.size(0), self.num_vars, -1)
            inp_n = self.input_n(ins)  # .view(inputs.size(0), self.num_vars, -1)
            # tmp = self.hidden_r(hidden_state)
            r = torch.sigmoid(inp_r + self.hidden_r(agg_msgs))
            i = torch.sigmoid(inp_i + self.hidden_i(agg_msgs))
            n = torch.tanh(inp_n + r * self.hidden_h(agg_msgs))
            hidden_state = ((1 - i) * n + i * hidden_state)
            hidden_seq.append(hidden_state)
        context = torch.stack(hidden_seq, dim=1)
        batch_size, seq_len, num_vertex, context_size = context.shape
        context_init = self.context_init[None, None, None, :].expand(batch_size, 1, num_vertex,
                                                                     -1)  # (batch_size, 1, context_size)
        # Shift the context by vectors by 1: context embedding after event i is used to predict event i + 1
        # if True:
        context = context[:, :-1, :, :]
        context = torch.cat([context_init, context], dim=1)
        # oh_marks = torch.nn.functional.one_hot(batch.marks, num_classes=4)
        # oh_marks = oh_marks[:,:,:,None].expand(batch_size,seq_len,num_vertex,hid_size)
        # context = oh_marks * context
        # context = torch.sum(context,axis = 2)
        # selected_context = context[:,:,batch.marks,:]
        ###
        context = context.view(context.size(0), context.size(1), -1)

        inter_time_dist = self.get_inter_time_dist(context)
        inter_times = batch.inter_times.clamp(1e-10)
        log_p = inter_time_dist.log_prob(inter_times)  # (batch_size, seq_len)

        # Survival probability of the last interval (from t_N to t_end).
        # You can comment this section of the code out if you don't want to implement the log_survival_function
        # for the distribution that you are using. This will make the likelihood computation slightly inaccurate,
        # but the difference shouldn't be significant if you are working with long sequences.
        last_event_idx = batch.mask.sum(-1, keepdim=True).long()  # (batch_size, 1)
        log_surv_all = inter_time_dist.log_survival_function(inter_times)  # (batch_size, seq_len)
        log_surv_last = torch.gather(log_surv_all, dim=-1, index=last_event_idx).squeeze(-1)  # (batch_size,)

        if self.num_marks > 1:
            mark_logits = torch.log_softmax(self.mark_linear(context), dim=-1)  # (batch_size, seq_len, num_marks)
            mark_dist = Categorical(logits=mark_logits)
            log_p += mark_dist.log_prob(batch.marks)  # (batch_size, seq_len)
        log_p *= batch.mask  # (batch_size, seq_len)
        return log_p.sum(-1) + log_surv_last  # (batch_size,)

    def node2edge(self, node_embeddings):
        send_embed = node_embeddings[:, self.send_edges, :]
        recv_embed = node_embeddings[:, self.recv_edges, :]
        return torch.cat([send_embed, recv_embed], dim=2)

    def node2edge_v1(self, node_embeddings):
        send_embed = node_embeddings[:, self.send_edges, :]
        recv_embed = node_embeddings[:, self.recv_edges, :]
        return torch.cat([send_embed, recv_embed], dim=3)

    def edge2node(self, edge_embeddings):
        incoming = torch.matmul(self.node2edge_mat, edge_embeddings)
        return incoming/(self.dimension-1) #TODO: do we want this average?

    def edge2node_v1(self, edge_embeddings):
        if len(edge_embeddings.shape) == 4:
            old_shape = edge_embeddings.shape
            tmp_embeddings = edge_embeddings.view(old_shape[0], old_shape[1], -1)
            incoming = torch.matmul(self.node2edge_mat, tmp_embeddings).view(old_shape[0], -1, old_shape[2], old_shape[3])
        else:
            incoming = torch.matmul(self.node2edge_mat, edge_embeddings)
        return incoming/(self.dimension-1)

    def log_prob_with_est_graph_v1(self, batch: dpp.data.Batch, mode) -> torch.Tensor:

        features = self.get_features(batch)
        ndim = self.dimension
        numbatch = batch.inter_times.size(0)
        # expanded_features = torch.zeros(batch.inter_times.size(0), ndim, batch.inter_times.size(1))
        expanded_features = torch.zeros(batch.inter_times.size(0), ndim, self.dimension_len)
        for ibatch in range(numbatch):
            for i in range(ndim):
                ind = torch.where(batch.marks[ibatch,
                                  :, ] == i)  # [idx for idx, element in enumerate(mixseq_marks) if mixseq_marks==i]
                expanded_features[ibatch, i, ind[0]] = batch.inter_times[ibatch, ind[0]]

        ###
        rnn_input = expanded_features
        ######################################
        ## encoder
        ######################################
        x = rnn_input
        x = self.mlp1(x)  # 2-layer ELU net per node

        x = self.node2edge(x)
        x = self.mlp2(x)
        x_skip = x

        if self.factor:
            x = self.edge2node(x)
            x = self.mlp3(x)
            x = self.node2edge(x)
            x = torch.cat((x, x_skip), dim=-1)  # Skip connection
            x = self.mlp4(x)
        else:
            x = self.mlp3(x)
            x = torch.cat((x, x_skip), dim=-1)  # Skip connection
            x = self.mlp4(x)
        logits = self.fc_out(x)

        old_shape = logits.shape
        edges = gumbel_softmax(
            logits.view(-1, 2),
            tau=self.gumbel_temp,
            hard= False).view(old_shape)
        self.rel_type = edges

        prob = F.softmax(logits, dim=-1)
        loss_kl = self.kl_categorical(prob)

        # rel_type = torch.zeros((self.dimension * (self.dimension - 1), self.num_edges))
        # self.rel_type = rel_type
        ######################################
        ## decoder
        ######################################
        time_steps = rnn_input.size(2)
        hid_size = self.context_size#150#64 * 2
        hidden_state = torch.zeros(rnn_input.size(0), rnn_input.size(1), hid_size)
        hidden_seq = []
        for istep in range(time_steps):
            #####
            # node2edge
            receivers = hidden_state[:, self.recv_edges, :]
            senders = hidden_state[:, self.send_edges, :]

            pre_msg = torch.cat([receivers, senders], dim=-1)

            # if inputs.is_cuda:
            #     all_msgs = torch.cuda.FloatTensor(pre_msg.size(0), pre_msg.size(1),
            #                                       self.msg_out_shape).fill_(0.)
            # else:
            all_msgs = torch.zeros(pre_msg.size(0), pre_msg.size(1),
                                   self.msg_out_shape)

            if self.skip_first_edge_type:
                start_idx = 1
                norm = float(len(self.msg_fc2)) - 1
            else:
                start_idx = 0
                norm = float(len(self.msg_fc2))

            # Run separate MLP for every edge type
            # NOTE: to exclude one edge type, simply offset range by 1
            # tmp = len(self.msg_fc2)
            # rel_type = self.rel_type.unsqueeze(0).expand(pre_msg.size(0),self.rel_type.size(0),self.rel_type.size(1))
            for i in range(start_idx, len(self.msg_fc2)):
                msg = torch.tanh(self.msg_fc1[i](pre_msg))
                msg = F.dropout(msg, p=self.dropout_prob)
                msg = torch.tanh(self.msg_fc2[i](msg))
                msg = msg * self.rel_type[:, :, i:i + 1]
                all_msgs += msg / norm

            # This step sums all of the messages per node
            agg_msgs = all_msgs.transpose(-2, -1).matmul(self.edge2node_mat).transpose(-2, -1)
            agg_msgs = agg_msgs.contiguous() / (self.num_nodes - 1)  # Average
            #####
            ins = rnn_input[:, :, istep].unsqueeze(-1)
            inp_r = self.input_r(ins)  # .view(inputs.size(0), self.num_vars, -1)
            inp_i = self.input_i(ins)  # .view(inputs.size(0), self.num_vars, -1)
            inp_n = self.input_n(ins)  # .view(inputs.size(0), self.num_vars, -1)
            r = torch.sigmoid(inp_r + self.hidden_r(agg_msgs))
            i = torch.sigmoid(inp_i + self.hidden_i(agg_msgs))
            n = torch.tanh(inp_n + r * self.hidden_h(agg_msgs))
            hidden_state = ((1 - i) * n + i * hidden_state)
            hidden_seq.append(hidden_state)

        context = torch.stack(hidden_seq, dim=1)
        batch_size, seq_len, num_vertex, context_size = context.shape
        context_init = self.context_init[None, None, None, :].expand(batch_size, 1, num_vertex, -1)  # (batch_size, 1, context_size)
        # Shift the context by vectors by 1: context embedding after event i is used to predict event i + 1
        # if True:
        context = context[:, :-1, :, :]
        context = torch.cat([context_init, context], dim=1)
        ###
        context = context.view(context.size(0), context.size(1), -1)

        inter_time_dist = self.get_inter_time_dist(context)
        inter_times = batch.inter_times#.clamp(1e-10)
        supp = torch.zeros(inter_times.size(0), self.dimension_len-inter_times.size(1))
        augment_mask = torch.cat((batch.mask,supp),1)
        inter_times = torch.cat((inter_times,supp),1)
        inter_times = inter_times.clamp(1e-10)
        log_p = inter_time_dist.log_prob(inter_times)  # (batch_size, seq_len)

        # Survival probability of the last interval (from t_N to t_end).
        # You can comment this section of the code out if you don't want to implement the log_survival_function
        # for the distribution that you are using. This will make the likelihood computation slightly inaccurate,
        # but the difference shouldn't be significant if you are working with long sequences.
        last_event_idx = batch.mask.sum(-1, keepdim=True).long()  # (batch_size, 1)
        log_surv_all = inter_time_dist.log_survival_function(inter_times)  # (batch_size, seq_len)
        log_surv_last = torch.gather(log_surv_all, dim=-1, index=last_event_idx).squeeze(-1)  # (batch_size,)

        if self.num_marks > 1:
            mark_logits = torch.log_softmax(self.mark_linear(context), dim=-1)  # (batch_size, seq_len, num_marks)
            mark_dist = Categorical(logits=mark_logits)
            log_p += mark_dist.log_prob(batch.marks)  # (batch_size, seq_len)
        log_p *= augment_mask  # (batch_size, seq_len)

        return log_p.sum(-1) + log_surv_last + 0.5 * loss_kl# , log_p.sum(-1) + log_surv_last  # (batch_size,)

    def log_prob_with_dynamic_graph(self, batch: dpp.data.Batch) -> torch.Tensor:
        ######################################
        ## shape input
        ######################################
        features = self.get_features(batch)
        ndim = self.dimension
        numbatch = batch.inter_times.size(0)
        # rnn_input = torch.zeros(batch.inter_times.size(0), ndim, batch.inter_times.size(1))
        rnn_input = torch.zeros(batch.inter_times.size(0), ndim, self.dimension_len)
        for ibatch in range(numbatch):
            for i in range(ndim):
                ind = torch.where(batch.marks[ibatch,
                                  :, ] == i)  # [idx for idx, element in enumerate(mixseq_marks) if mixseq_marks==i]
                rnn_input[ibatch, i, ind[0]] = batch.inter_times[ibatch, ind[0]]
        # rnn_input = expanded_features
        ######################################
        ## encoder
        ######################################
        # x = rnn_input.transpose(1, 2).contiguous().view(rnn_input.size(0), rnn_input.size(2), -1)
        # x = rnn_input
        x = rnn_input.unsqueeze(-1)
        # New shape: [num_sims, num_atoms, num_timesteps*num_dims]
        x = self.mlp1_v1(x)  # 2-layer ELU net per node
        x = self.node2edge_v1(x)
        x = self.mlp2(x)
        x_skip = x

        if self.factor:
            x = self.edge2node_v1(x)
            x = self.mlp3(x)
            x = self.node2edge_v1(x)
            x = torch.cat((x, x_skip), dim=-1)  # Skip connection
            x = self.mlp4(x)
        else:
            x = self.mlp3(x)
            x = torch.cat((x, x_skip), dim=-1)  # Skip connection
            x = self.mlp4(x)

        old_shape = x.shape
        x = x.contiguous().view(-1, old_shape[2], old_shape[3])
        forward_x, prior_state = self.forward_rnn(x)
        timesteps = old_shape[2]
        reverse_x = x.flip(1)
        reverse_x, _ = self.reverse_rnn(reverse_x)
        reverse_x = reverse_x.flip(1)

        # x: [batch*num_edges, num_timesteps, hidden_size]
        # prior_result = self.prior_fc_out(forward_x).view(old_shape[0], old_shape[1], timesteps,
        #                                                  self.num_edges).transpose(1, 2).contiguous()
        combined_x = torch.cat([forward_x, reverse_x], dim=-1)
        reformed_x = combined_x.view(combined_x.size(0),-1)
        pl = self.encoder_fc_out_v1(reformed_x)
        pl = pl.view(old_shape[0],old_shape[1],self.num_prd,self.num_edges)
        posterior_logits = pl.transpose(1, 2).contiguous()

        all_edges = torch.zeros(posterior_logits.size())
        for istep in range(self.num_prd):
            current_p_logits = posterior_logits[:, istep]
            old_shape = current_p_logits.shape
            hard_sample = False
            all_edges[:,istep] = gumbel_softmax(
                current_p_logits.reshape(-1, 2),
                tau=self.gumbel_temp,
                hard=hard_sample).view(old_shape)

        ######################################
        ## decoder
        ######################################
        time_steps = batch.inter_times.size(1)
        # print(time_steps)
        hid_size = self.context_size#150#64 * 2
        hidden_state = torch.zeros(rnn_input.size(0), rnn_input.size(1), hid_size)
        hidden_seq = []
        for istep in range(time_steps):
            edges = all_edges[torch.arange(0,10,1),batch.time_interval_idx[:,istep],:,:]

            # node2edge
            receivers = hidden_state[:, self.recv_edges, :]
            senders = hidden_state[:, self.send_edges, :]

            # pre_msg: [batch, num_edges, 2*msg_out]
            pre_msg = torch.cat([receivers, senders], dim=-1)

            # if inputs.is_cuda:
            #     all_msgs = torch.cuda.FloatTensor(pre_msg.size(0), pre_msg.size(1),
            #                                       self.msg_out_shape).fill_(0.)
            # else:
            all_msgs = torch.zeros(pre_msg.size(0), pre_msg.size(1),
                                   self.msg_out_shape)

            if self.skip_first_edge_type:
                start_idx = 1
                norm = float(len(self.msg_fc2)) - 1
            else:
                start_idx = 0
                norm = float(len(self.msg_fc2))

            # Run separate MLP for every edge type
            # NOTE: to exclude one edge type, simply offset range by 1
            # tmp = len(self.msg_fc2)
            # rel_type = self.rel_type.unsqueeze(0).expand(pre_msg.size(0),self.rel_type.size(0),self.rel_type.size(1))
            for i in range(start_idx, len(self.msg_fc2)):
                msg = torch.tanh(self.msg_fc1[i](pre_msg))
                msg = F.dropout(msg, p=self.dropout_prob)
                msg = torch.tanh(self.msg_fc2[i](msg))
                msg = msg * edges[:, :, i:i + 1]
                all_msgs += msg / norm

            # This step sums all of the messages per node
            agg_msgs = all_msgs.transpose(-2, -1).matmul(self.edge2node_mat).transpose(-2, -1)
            agg_msgs = agg_msgs.contiguous() / (self.num_nodes - 1)  # Average
            #####
            ins = rnn_input[:, :, istep].unsqueeze(-1)
            inp_r = self.input_r(ins)  # .view(inputs.size(0), self.num_vars, -1)
            inp_i = self.input_i(ins)  # .view(inputs.size(0), self.num_vars, -1)
            inp_n = self.input_n(ins)  # .view(inputs.size(0), self.num_vars, -1)
            # tmp = self.hidden_r(hidden_state)
            r = torch.sigmoid(inp_r + self.hidden_r(agg_msgs))
            i = torch.sigmoid(inp_i + self.hidden_i(agg_msgs))
            n = torch.tanh(inp_n + r * self.hidden_h(agg_msgs))
            hidden_state = ((1 - i) * n + i * hidden_state)
            hidden_seq.append(hidden_state)
        context = torch.stack(hidden_seq, dim=1)
        batch_size, seq_len, num_vertex, context_size = context.shape
        context_init = self.context_init[None, None, None, :].expand(batch_size, 1, num_vertex, -1)  # (batch_size, 1, context_size)
        # Shift the context by vectors by 1: context embedding after event i is used to predict event i + 1
        # if True:
        context = context[:, :-1, :, :]
        context = torch.cat([context_init, context], dim=1)

        context = context.view(context.size(0), context.size(1), -1)

        inter_time_dist = self.get_inter_time_dist(context)
        inter_times = batch.inter_times.clamp(1e-10)
        # supp = torch.zeros(inter_times.size(0), self.dimension_len - inter_times.size(1))
        # augment_mask = torch.cat((batch.mask, supp), 1)
        # inter_times = torch.cat((inter_times, supp), 1)
        # inter_times = inter_times.clamp(1e-10)
        log_p = inter_time_dist.log_prob(inter_times)  # (batch_size, seq_len)

        last_event_idx = batch.mask.sum(-1, keepdim=True).long()  # (batch_size, 1)
        log_surv_all = inter_time_dist.log_survival_function(inter_times)  # (batch_size, seq_len)
        log_surv_last = torch.gather(log_surv_all, dim=-1, index=last_event_idx).squeeze(-1)  # (batch_size,)

        if self.num_marks > 1:
            mark_logits = torch.log_softmax(self.mark_linear(context), dim=-1)  # (batch_size, seq_len, num_marks)
            mark_dist = Categorical(logits=mark_logits)
            log_p += mark_dist.log_prob(batch.marks)  # (batch_size, seq_len)
        log_p *= batch.mask  # (batch_size, seq_len)
        return log_p.sum(-1) + log_surv_last  # (batch_size,)

    def log_prob_with_dynamic_graph_v1(self, batch: dpp.data.Batch, mode, epoch, filename) -> torch.Tensor:
        ######################################
        features = self.get_features(batch)
        ######################################
        ndim = self.dimension
        numbatch = batch.inter_times.size(0)
        # rnn_input = torch.zeros(batch.inter_times.size(0), ndim, batch.inter_times.size(1))
        start = timeit.default_timer()
        rnn_input = torch.zeros(batch.inter_times.size(0), ndim, self.dimension_len)
        for ibatch in range(numbatch):
            for i in range(ndim):
                ind = torch.where(batch.marks[ibatch,
                                  :, ] == i)  # [idx for idx, element in enumerate(mixseq_marks) if mixseq_marks==i]
                rnn_input[ibatch, i, ind[0]] = batch.inter_times[ibatch, ind[0]]
        # rnn_input = expanded_features
        # stop = timeit.default_timer()
        # print('Time: ', stop - start)

        ######################################
        ## encoder
        ######################################
        # x = rnn_input.transpose(1, 2).contiguous().view(rnn_input.size(0), rnn_input.size(2), -1)
        # x = rnn_input
        x = rnn_input.unsqueeze(-1)

        # New shape: [num_sims, num_atoms, num_timesteps*num_dims]
        x = self.mlp1_v1(x)  # 2-layer ELU net per node
        x = self.node2edge_v1(x)
        x = self.mlp2(x)
        # x_skip = x
        #
        # if self.factor:
        #     x = self.edge2node_v1(x)
        #     x = self.mlp3(x)
        #     x = self.node2edge_v1(x)
        #     x = torch.cat((x, x_skip), dim=-1)  # Skip connection
        #     x = self.mlp4(x)
        # else:
        #     x = self.mlp3(x)
        #     x = torch.cat((x, x_skip), dim=-1)  # Skip connection
        #     x = self.mlp4(x)

        old_shape = x.shape
        x = x.contiguous().view(-1, old_shape[2], old_shape[3])
        x_compressed = self.compressor(x.transpose(1,2)).transpose(1,2)
        forward_x, prior_state = self.forward_rnn(x_compressed)
        timesteps = old_shape[2]
        reverse_x = x_compressed.flip(1)
        reverse_x, _ = self.reverse_rnn(reverse_x)
        reverse_x = reverse_x.flip(1)

        # x: [batch*num_edges, num_timesteps, hidden_size]
        prior_logits = self.prior_fc_out(forward_x).view(old_shape[0], old_shape[1], self.num_prd,
                                                         self.num_edges).transpose(1, 2).contiguous()
        combined_x = torch.cat([forward_x, reverse_x], dim=-1)
        # reformed_x = combined_x.view(combined_x.size(0),-1)
        pl = self.encoder_fc_out(combined_x)
        pl = pl.view(old_shape[0],old_shape[1],self.num_prd,self.num_edges)
        posterior_logits = pl.transpose(1, 2).contiguous()

        all_edges = torch.zeros(posterior_logits.size())

        for istep in range(self.num_prd):
            current_p_logits = posterior_logits[:, istep]
            # current_p_logits = prior_logits[:, istep]
            old_shape = current_p_logits.shape
            hard_sample = False
            all_edges[:,istep] = gumbel_softmax(
                current_p_logits.reshape(-1, 2),
                tau=self.gumbel_temp,
                hard=hard_sample).view(old_shape)

        prob = F.softmax(posterior_logits, dim=-1)
        loss_kl = self.kl_categorical_learned(prob, prior_logits)

        if epoch%50==00:
            torch.save(all_edges, filename + '-edges-epoch'+str(epoch)+'.pt')

        # stop = timeit.default_timer()
        # print('Time: ', stop - start)



        # pl = self.encoder_fc_out(combined_x)
        # posterior_logits = pl.view(old_shape[0], old_shape[1], timesteps, self.num_edges).transpose(1, 2).contiguous()

        # logits = self.fc_out(x)
        # # result_dict = {
        # #     'logits': result,
        # #     'state': features,
        # # }
        # old_shape = logits.shape
        # edges = gumbel_softmax(
        #     logits.view(-1, 2),
        #     tau=self.gumbel_temp,
        #     hard= True).view(old_shape)
        # self.rel_type = edges
        ######################################
        ## decoder
        ######################################
        # time_interval_idx = torch.cat((batch.time_interval_idx,supp),1)
        # time_steps = rnn_input.size(2)
        time_steps = batch.inter_times.size(1)
        # print(time_steps)
        hid_size = self.context_size#150#64 * 2
        hidden_state = torch.zeros(rnn_input.size(0), rnn_input.size(1), hid_size)
        hidden_seq = []
        for istep in range(time_steps):
            # current_p_logits = posterior_logits[:, istep]
            # old_shape = current_p_logits.shape
            # hard_sample = False
            # edge_idx = torch.floor(batch.arr_times[:,istep]/3600).long()
            # if istep < batch.time_interval_idx.size(1):
            # if torch.sum(batch.time_interval_idx[:, istep]>30):
            #     print('d')
            # xx = torch.arange(0, batch.size, 1)
            # yy = batch.time_interval_idx[:, istep]
            edges = all_edges[torch.arange(0, batch.size, 1), batch.time_interval_idx[:, istep], :, :]

            # node2edge
            receivers = hidden_state[:, self.recv_edges, :]
            senders = hidden_state[:, self.send_edges, :]

            # pre_msg: [batch, num_edges, 2*msg_out]
            pre_msg = torch.cat([receivers, senders], dim=-1)

            # if inputs.is_cuda:
            #     all_msgs = torch.cuda.FloatTensor(pre_msg.size(0), pre_msg.size(1),
            #                                       self.msg_out_shape).fill_(0.)
            # else:
            all_msgs = torch.zeros(pre_msg.size(0), pre_msg.size(1),
                                   self.msg_out_shape)

            if self.skip_first_edge_type:
                start_idx = 1
                norm = float(len(self.msg_fc2)) - 1
            else:
                start_idx = 0
                norm = float(len(self.msg_fc2))

            # Run separate MLP for every edge type
            # NOTE: to exclude one edge type, simply offset range by 1
            # tmp = len(self.msg_fc2)
            # rel_type = self.rel_type.unsqueeze(0).expand(pre_msg.size(0),self.rel_type.size(0),self.rel_type.size(1))
            for i in range(start_idx, len(self.msg_fc2)):
                msg = torch.tanh(self.msg_fc1[i](pre_msg))
                msg = F.dropout(msg, p=self.dropout_prob)
                msg = torch.tanh(self.msg_fc2[i](msg))
                msg = msg * edges[:, :, i:i + 1]
                all_msgs += msg / norm

            # This step sums all of the messages per node
            agg_msgs = all_msgs.transpose(-2, -1).matmul(self.edge2node_mat).transpose(-2, -1)
            agg_msgs = agg_msgs.contiguous() / (self.num_nodes - 1)  # Average
            #####
            ins = rnn_input[:, :, istep].unsqueeze(-1)
            inp_r = self.input_r(ins)  # .view(inputs.size(0), self.num_vars, -1)
            inp_i = self.input_i(ins)  # .view(inputs.size(0), self.num_vars, -1)
            inp_n = self.input_n(ins)  # .view(inputs.size(0), self.num_vars, -1)
            # tmp = self.hidden_r(hidden_state)
            r = torch.sigmoid(inp_r + self.hidden_r(agg_msgs))
            i = torch.sigmoid(inp_i + self.hidden_i(agg_msgs))
            n = torch.tanh(inp_n + r * self.hidden_h(agg_msgs))
            hidden_state = ((1 - i) * n + i * hidden_state)
            hidden_seq.append(hidden_state)

        context = torch.stack(hidden_seq, dim=1)

        # stop = timeit.default_timer()
        # print('Time: ', stop - start)

        context0 = self.fwd(features)[0]
        batch_size, seq_len, num_vertex, context_size = context.shape
        ####
        # dropout_prob = 0.5
        # context = context.view(-1,context.size(-1))
        # pred = F.dropout(F.relu(self.out_fc1(context)), p=dropout_prob)
        # pred = F.dropout(F.relu(self.out_fc2(pred)), p=dropout_prob)
        # context = self.out_fc3(pred)
        # context = context.view(batch_size, seq_len, num_vertex, context_size)
        ####

        context_init = self.context_init[None, None, None, :].expand(batch_size, 1, num_vertex, -1)  # (batch_size, 1, context_size)
        # Shift the context by vectors by 1: context embedding after event i is used to predict event i + 1
        # if True:
        context = context[:, :-1, :, :]
        context = torch.cat([context_init, context], dim=1)

        context = context.view(context.size(0), context.size(1), -1)

        inter_time_dist = self.get_inter_time_dist(context)

        # pred_time = inter_time_dist.sample()
        # pred_time = self.sample(2,5)
        # pred_time = inter_time_dist.mean
        #
        inter_times = batch.inter_times.clamp(1e-10)
        #
        # # rmse = sqrt(mean_squared_error(pred_time,inter_times))
        # # print('RMSE:',rmse)
        # # print(se)

        # supp = torch.zeros(inter_times.size(0), self.dimension_len - inter_times.size(1))
        # augment_mask = torch.cat((batch.mask, supp), 1)
        # inter_times = torch.cat((inter_times, supp), 1)
        # inter_times = inter_times.clamp(1e-10)
        log_p = inter_time_dist.log_prob(inter_times)  # (batch_size, seq_len)

        last_event_idx = batch.mask.sum(-1, keepdim=True).long()  # (batch_size, 1)
        log_surv_all = inter_time_dist.log_survival_function(inter_times)  # (batch_size, seq_len)
        log_surv_last = torch.gather(log_surv_all, dim=-1, index=last_event_idx).squeeze(-1)  # (batch_size,)

        # if self.num_marks > 1:
        #     mark_logits = torch.log_softmax(self.mark_linear(context), dim=-1)  # (batch_size, seq_len, num_marks)
        #     mark_dist = Categorical(logits=mark_logits)
        #     log_p += mark_dist.log_prob(batch.marks)  # (batch_size, seq_len)

        log_p *= batch.mask  # (batch_size, seq_len)

        #########

        # pred_time = nn.relu(pred_time)
        # pred_time = nn.Linear(pred_time,)

        # context0 = features
        context = torch.cat([context, context0],dim=-1)

        # dropout_prob = 0.5
        # # pred_time = F.relu(self.time_pred1(context))
        # # pred = F.dropout(F.relu(self.time_pred1(context)), p=dropout_prob)
        # # pred = F.dropout(F.relu(self.time_pred2(pred)), p=dropout_prob)
        # pred_time = self.time_pred3(pred)
        pred_mark = self.mark_predictor(context)
        true_mark = batch.marks[:, 1:]-1
        pred_mark = pred_mark[:, :-1, :]

        prediction = torch.max(pred_mark, dim=-1)[1]
        correct_num = torch.sum(prediction == true_mark)

        loss_mark = self.loss_mark_func(pred_mark.float(), true_mark)
        loss_mark = torch.sum(loss_mark)
        ######################################################
        pred_time = self.time_predictor(context)
        pred_time = pred_time.squeeze_(-1)
        ##########
        for iii in range(len(last_event_idx)):
            inter_times[iii,last_event_idx] = 0
        inter_times = inter_times.clamp(1e-10)

        #true = inter_times[:, 1:]
        #pred_time = pred_time[:, :-1]
        true = inter_times[:, 1:]
        prediction = pred_time[:, :-1]

        # # event time gap prediction
        # diff = prediction - true
        # se = torch.sum(diff * diff)

        diff = pred_time - inter_times
        tmp = diff * diff
        rmse = torch.sum(tmp)

        return (log_p.sum(-1)  + log_surv_last - 0.25*loss_kl), rmse, loss_mark, correct_num

    def log_prob_with_dynamic_graph_v2(self, batch: dpp.data.Batch, mode) -> torch.Tensor:

        ######################################
        features = self.get_features(batch)

        # #####################################
        # ndim = self.dimension
        # numbatch = batch.inter_times.size(0)
        #
        # rnn_input = torch.zeros(batch.inter_times.size(0), ndim, self.dimension_len)
        # for ibatch in range(numbatch):
        #     for i in range(ndim):
        #         ind = torch.where(batch.marks[ibatch,
        #                           :, ] == i)  # [idx for idx, element in enumerate(mixseq_marks) if mixseq_marks==i]
        #         rnn_input[ibatch, i, ind[0]] = batch.inter_times[ibatch, ind[0]]
        #
        # ######################################
        # ## encoder
        # ######################################
        # x = rnn_input.unsqueeze(-1)
        # ## New shape: [num_sims, num_atoms, num_timesteps*num_dims]
        # x = self.mlp1_v1(x)  # 2-layer ELU net per node
        # x = self.node2edge_v1(x)
        # x = self.mlp2(x)
        # # x_skip = x
        #
        # # if self.factor:
        # #     x = self.edge2node_v1(x)
        # #     x = self.mlp3(x)
        # #     x = self.node2edge_v1(x)
        # #     x = torch.cat((x, x_skip), dim=-1)  # Skip connection
        # #     x = self.mlp4(x)
        # # else:
        # #     x = self.mlp3(x)
        # #     x = torch.cat((x, x_skip), dim=-1)  # Skip connection
        # #     x = self.mlp4(x)
        #
        # old_shape = x.shape
        # x = x.contiguous().view(-1, old_shape[2], old_shape[3])
        # x_compressed = self.compressor(x.transpose(1,2)).transpose(1,2)
        #
        # forward_x, prior_state = self.forward_rnn(x_compressed)
        # # timesteps = old_shape[2]
        # reverse_x = x_compressed.flip(1)
        # reverse_x, _ = self.reverse_rnn(reverse_x)
        # reverse_x = reverse_x.flip(1)
        #
        # # x: [batch*num_edges, num_timesteps, hidden_size]
        # prior_logits = self.prior_fc_out(forward_x).view(old_shape[0], old_shape[1], self.num_prd,
        #                                                  self.num_edges).transpose(1, 2).contiguous()
        #
        # combined_x = torch.cat([forward_x, reverse_x], dim=-1)
        # # reformed_x = combined_x.view(combined_x.size(0),-1)
        # pl = self.encoder_fc_out(combined_x)
        # pl = pl.view(old_shape[0],old_shape[1],self.num_prd,self.num_edges)
        # posterior_logits = pl.transpose(1, 2).contiguous()
        #
        # all_edges = torch.zeros(posterior_logits.size())
        # for istep in range(self.num_prd):
        #     current_p_logits = posterior_logits[:, istep]
        #     old_shape = current_p_logits.shape
        #     hard_sample = False
        #     all_edges[:,istep] = gumbel_softmax(
        #         current_p_logits.reshape(-1, 2),
        #         tau=self.gumbel_temp,
        #         hard=hard_sample).view(old_shape)
        #
        # prob = F.softmax(posterior_logits, dim=-1)
        # loss_kl = self.kl_categorical_learned(prob, prior_logits)
        #
        # time_steps = batch.inter_times.size(1)
        #
        # hid_size = self.context_size#150#64 * 2
        # hidden_state = torch.zeros(rnn_input.size(0), rnn_input.size(1), hid_size)
        # hidden_seq = []
        # start_idx = 1
        #
        # for istep in range(time_steps):
        #
        #     edges = all_edges[torch.arange(0, batch.size, 1), batch.time_interval_idx[:, istep], :, :]
        #
        #     receivers = hidden_state[:, self.recv_edges, :]
        #     senders = hidden_state[:, self.send_edges, :]
        #
        #     pre_msg = torch.cat([receivers, senders], dim=-1)
        #
        #     all_msgs = torch.zeros(pre_msg.size(0), pre_msg.size(1),
        #                            self.msg_out_shape)
        #
        #     for i in range(start_idx, len(self.msg_fc2)):
        #         msg = torch.tanh(self.msg_fc1[i](pre_msg))
        #         msg = F.dropout(msg, p=self.dropout_prob)
        #         msg = torch.tanh(self.msg_fc2[i](msg))
        #         msg = msg * edges[:, :, i:i + 1]
        #         all_msgs += msg# / norm
        #
        #     # This step sums all of the messages per node
        #     agg_msgs = all_msgs.transpose(-2, -1).matmul(self.edge2node_mat).transpose(-2, -1)
        #     agg_msgs = agg_msgs.contiguous() / (self.num_nodes - 1)  # Average
        #
        #     #####
        #     ins = rnn_input[:, :, istep].unsqueeze(-1)
        #     inp_r = self.input_r(ins)  # .view(inputs.size(0), self.num_vars, -1)
        #     inp_i = self.input_i(ins)  # .view(inputs.size(0), self.num_vars, -1)
        #     inp_n = self.input_n(ins)  # .view(inputs.size(0), self.num_vars, -1)
        #     # tmp = self.hidden_r(hidden_state)
        #     r = torch.sigmoid(inp_r + self.hidden_r(agg_msgs))
        #     i = torch.sigmoid(inp_i + self.hidden_i(agg_msgs))
        #     n = torch.tanh(inp_n + r * self.hidden_h(agg_msgs))
        #     hidden_state = ((1 - i) * n + i * hidden_state)
        #     hidden_seq.append(hidden_state)
        #
        # context = torch.stack(hidden_seq, dim=1)

        context0 = self.fwd(features)[0]
        ####
        batch_size, seq_len, context_size = context0.shape
        context = torch.zeros((batch_size, seq_len, self.num_nodes, context_size))
        context = context.view(context.size(0), context.size(1), -1)
        ####
        # batch_size, seq_len, num_vertex, context_size = context.shape
        # ####
        #
        # context_init = self.context_init[None, None, None, :].expand(batch_size, 1, num_vertex, -1)  # (batch_size, 1, context_size)
        # # Shift the context by vectors by 1: context embedding after event i is used to predict event i + 1
        # # if True:
        # context = context[:, :-1, :, :]
        # context = torch.cat([context_init, context], dim=1)
        #
        # context = context.view(context.size(0), context.size(1), -1)

        inter_time_dist = self.get_inter_time_dist(context)
        #
        inter_times = batch.inter_times.clamp(1e-10)
        #
        log_p = inter_time_dist.log_prob(inter_times)  # (batch_size, seq_len)
        #
        last_event_idx = batch.mask.sum(-1, keepdim=True).long()  # (batch_size, 1)
        log_surv_all = inter_time_dist.log_survival_function(inter_times)  # (batch_size, seq_len)
        log_surv_last = torch.gather(log_surv_all, dim=-1, index=last_event_idx).squeeze(-1)  # (batch_size,)

        log_p *= batch.mask  # (batch_size, seq_len)

        #########
        context = torch.cat([context, context0],dim=-1)

        pred_mark = self.mark_predictor(context)
        true_mark = batch.marks[:, 1:]-1
        pred_mark = pred_mark[:, :-1, :]

        prediction = torch.max(pred_mark, dim=-1)[1]
        correct_num = torch.sum(prediction == true_mark)

        loss_mark = self.loss_mark_func(pred_mark.float(), true_mark)
        loss_mark = torch.sum(loss_mark)
        ######################################################
        pred_time = self.time_predictor(context)
        pred_time = pred_time.squeeze_(-1)
        ##########
        for iii in range(len(last_event_idx)):
            inter_times[iii,last_event_idx] = 0
        inter_times = inter_times.clamp(1e-10)

        true = inter_times[:, 1:]
        prediction = pred_time[:, :-1]

        diff = pred_time - inter_times
        tmp = diff * diff
        rmse = torch.sum(tmp)

        return (log_p.sum(-1) + log_surv_last - 0.25*loss_kl), rmse, loss_mark, correct_num
        # return torch.tensor(0, dtype=torch.int8), rmse, loss_mark, correct_num

    def sample(self, t_end: float, batch_size: int = 1, context_init: torch.Tensor = None) -> dpp.data.Batch:
        """Generate a batch of sequence from the model.

        Args:
            t_end: Size of the interval on which to simulate the TPP.
            batch_size: Number of independent sequences to simulate.
            context_init: Context vector for the first event.
                Can be used to condition the generator on past events,
                shape (context_size,)

        Returns;
            batch: Batch of sampled sequences. See dpp.data.batch.Batch.
        """
        if context_init is None:
            # Use the default context vector
            context_init = self.context_init
        else:
            # Use the provided context vector
            context_init = context_init.view(self.context_size)
        next_context = context_init[None, None, :].expand(batch_size, 1, -1)
        inter_times = torch.empty(batch_size, 0)
        if self.num_marks > 1:
            marks = torch.empty(batch_size, 0, dtype=torch.long)

        generated = False
        while not generated:
            inter_time_dist = self.get_inter_time_dist(next_context)
            next_inter_times = inter_time_dist.sample()  # (batch_size, 1)
            inter_times = torch.cat([inter_times, next_inter_times], dim=1)  # (batch_size, seq_len)

            # Generate marks, if necessary
            if self.num_marks > 1:
                mark_logits = torch.log_softmax(self.mark_linear(next_context), dim=-1)  # (batch_size, 1, num_marks)
                mark_dist = Categorical(logits=mark_logits)
                next_marks = mark_dist.sample()  # (batch_size, 1)
                marks = torch.cat([marks, next_marks], dim=1)
            else:
                marks = None

            with torch.no_grad():
                generated = inter_times.sum(-1).min() >= t_end
            batch = Batch(inter_times=inter_times, mask=torch.ones_like(inter_times), marks=marks)
            features = self.get_features(batch)  # (batch_size, seq_len, num_features)
            context = self.get_context(features, remove_last=False)  # (batch_size, seq_len, context_size)
            next_context = context[:, [-1], :]  # (batch_size, 1, context_size)

        arrival_times = inter_times.cumsum(-1)  # (batch_size, seq_len)
        inter_times = diff(arrival_times.clamp(max=t_end), dim=-1)
        mask = (arrival_times <= t_end).float()  # (batch_size, seq_len)
        if self.num_marks > 1:
            marks = marks * mask  # (batch_size, seq_len)
        return Batch(inter_times=inter_times, mask=mask, marks=marks)
