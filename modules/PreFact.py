import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_mean, scatter_softmax
import os
from logging import getLogger

logger = getLogger()

class AttentionPreferenceMiner(nn.Module):
    def __init__(self, num_users, num_items, emb_dim, dropout=0.1):
        super().__init__()
        self.user_embed = nn.Parameter(torch.empty(num_users, emb_dim))
        self.item_embed = nn.Parameter(torch.empty(num_items, emb_dim))
        nn.init.xavier_uniform_(self.user_embed)
        nn.init.xavier_uniform_(self.item_embed)

        self.query_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.key_proj   = nn.Linear(emb_dim, emb_dim, bias=False)
        self.value_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.dropout    = nn.Dropout(p=dropout)

    def forward(self, inter_edge, edge_weight=None):
        users, items = inter_edge[0, :], inter_edge[1, :]
        query = self.query_proj(self.user_embed)[users]  # [n_inter, emb_dim]
        key   = self.key_proj(self.item_embed)[items]    # [n_inter, emb_dim]
        value = self.value_proj(self.item_embed)[items]  # [n_inter, emb_dim]

        scores = (query * key).sum(dim=-1) / (self.user_embed.size(1) ** 0.5)
        if edge_weight is not None:
            scores = scores * edge_weight

        alpha = scatter_softmax(scores, index=users, dim=0)
        alpha = self.dropout(alpha)
        weighted_value = value * alpha.unsqueeze(-1)

        user_aggregated = scatter_mean(weighted_value, index=users, dim=0,
                                      dim_size=self.user_embed.size(0))
        return user_aggregated      

class FactSemanticEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.weight_predictor = nn.Sequential(
            nn.Linear(embedding_dim * 6, embedding_dim * 2),
            nn.Sigmoid(),
            nn.Linear(embedding_dim * 2, 3),
            nn.Softmax(dim=-1),
            nn.Dropout(p=0.1)
        )

    def forward(self, edge_index, edge_type, entity_embeddings, relation_embeddings):
        head_indices = edge_index[0, :]
        rel_indices = edge_type - 1
        tail_indices = edge_index[1, :]

        h_embs = entity_embeddings[head_indices]
        r_embs = relation_embeddings[rel_indices]
        t_embs = entity_embeddings[tail_indices]

        h_real, h_imag = h_embs.real, h_embs.imag
        r_real, r_imag = r_embs.real, r_embs.imag
        t_real, t_imag = t_embs.real, t_embs.imag

        combined_features = torch.cat([h_real, h_imag, r_real, r_imag, t_real, t_imag], dim=-1)
        weights = self.weight_predictor(combined_features)
        w_h = weights[:, 0:1]
        w_r = weights[:, 1:2]
        w_t = weights[:, 2:3]
        base_real = w_h * h_real + w_r * r_real + w_t * t_real
        base_imag = w_h * h_imag + w_r * r_imag + w_t * t_imag

        final_real = base_real
        final_imag = base_imag

        fact_embeddings = torch.cat([final_real, final_imag], dim=1)  # [n_triplets, emb_size]

        return fact_embeddings


class GatingRouter(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(p=0.1)
        )
        self.scale = hidden_dim ** 0.5
        self.batch_size = 4096

    def forward(self, user_preference, fact_embeddings):
        n_triplets = fact_embeddings.shape[0]
        omega = torch.zeros(n_triplets, device=fact_embeddings.device)

        for start in range(0, n_triplets, self.batch_size):
            end = min(start + self.batch_size, n_triplets)
            batch_fact = fact_embeddings[start:end]  # [batch_size, embed_dim]

            aligned_fact = self.mlp(batch_fact)  # [batch_size, embed_dim]


            batch_similarity = torch.matmul(aligned_fact, user_preference.T)
            omega[start:end] = torch.sum(batch_similarity, dim=1) / self.scale # [batch_size]

        return omega


def relation_aware_edge_sampling(edge_index, edge_type, omega, n_relations, samp_rate=0.5):
    # exclude interaction
    for i in range(1, n_relations):
        edge_index_i, edge_type_i, omega_i = edge_sampling(
            edge_index[:, edge_type == i], edge_type[edge_type == i], omega[edge_type == i], samp_rate)
        if i == 1:
            edge_index_sampled = edge_index_i
            edge_type_sampled = edge_type_i
            omega_sampled = omega_i
        else:
            edge_index_sampled = torch.cat(
                [edge_index_sampled, edge_index_i], dim=1)
            edge_type_sampled = torch.cat(
                [edge_type_sampled, edge_type_i], dim=0)
            omega_sampled = torch.cat(
                [omega_sampled, omega_i], dim=0)
    return edge_index_sampled, edge_type_sampled, omega_sampled


def edge_sampling(edge_index, edge_type, omega, samp_rate=0.5):
    # edge_index: [2, -1]
    # edge_type: [-1]
    n_edges = edge_index.shape[1]
    random_indices = np.random.choice(
        n_edges, size=int(n_edges * samp_rate), replace=False)
    return edge_index[:, random_indices], edge_type[random_indices], omega[random_indices]


def sparse_dropout(i, v, keep_rate=0.5):
    noise_shape = i.shape[1]

    random_tensor = keep_rate
    # the drop rate is 1 - keep_rate
    random_tensor += torch.rand(noise_shape).to(i.device)
    dropout_mask = torch.floor(random_tensor).type(torch.bool)

    i = i[:, dropout_mask]
    v = v[dropout_mask] / keep_rate

    return i, v


class GraphConv(nn.Module):
    def __init__(self, emb_size, n_hops, n_relations, mess_dropout_rate=0.1, gamma=0.2, N=3):
        super().__init__()
        self.gamma = gamma
        self.N = N

        initializer = nn.init.xavier_uniform_
        relation_emb = initializer(torch.empty(n_relations - 1, emb_size))  # not include interact
        self.relation_emb = nn.Parameter(relation_emb)  # [n_relations - 1, in_channel]

        self.n_hops = n_hops

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def aggregator(self, user_emb, entity_emb, edge_index, edge_type, omega, inter_edge, inter_edge_w, relation_emb, hop, gamma=None):
        n_entities = entity_emb.shape[0]
        head, tail = edge_index

        relation_emb = relation_emb[edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        neigh_relation_emb = entity_emb[tail] * relation_emb  # [-1, channel]

        alpha = omega / (scatter_sum(src=omega, index=head, dim_size=omega.shape[0], dim=0).index_select(0, head) + 1e-8)
        judge = alpha > gamma
        eta = judge.float() * alpha
        eta = eta / (scatter_sum(src=eta, index=head, dim_size=eta.shape[0], dim=0).index_select(0, head) + 1e-8)
        rho = alpha if hop < self.N else eta
        entity_agg = rho.unsqueeze(-1) * neigh_relation_emb

        entity_agg = scatter_mean(src=entity_agg, index=head, dim_size=n_entities, dim=0)

        item_agg = inter_edge_w.unsqueeze(-1) * entity_emb[inter_edge[1, :]]

        user_agg = scatter_sum(src=item_agg, index=inter_edge[0, :], dim_size=user_emb.shape[0], dim=0)
        return entity_agg, user_agg

    def forward(self, user_emb, entity_emb, edge_index, edge_type, omega,
                inter_edge, inter_edge_w, mess_dropout=True, gamma=None):
        entity_res_emb = entity_emb  # [n_entity, channel]
        user_res_emb = user_emb  # [n_users, channel]

        for i in range(self.n_hops):
            entity_emb, user_emb = self.aggregator(user_emb, entity_emb, edge_index, edge_type, omega,
                                                   inter_edge, inter_edge_w, self.relation_emb, i + 1, gamma=gamma)

            """message dropout"""
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
                user_emb = self.dropout(user_emb)
            entity_emb = F.normalize(entity_emb, dim=-1, eps=1e-8)
            user_emb = F.normalize(user_emb, dim=-1, eps=1e-8)
            entity_emb = torch.nan_to_num(entity_emb, nan=0.0, posinf=1e4, neginf=1e-4)
            user_emb = torch.nan_to_num(user_emb, nan=0.0, posinf=1e4, neginf=1e-4)

            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)

        return entity_res_emb, user_res_emb


class PreFact(nn.Module):
    def __init__(self, data_config, args_config, graph, adj_mat):
        super().__init__()
        self.args_config = args_config

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities

        self.margin_ccl = args_config.margin
        self.num_neg_sample = args_config.num_neg_sample
        self.decay = args_config.l2
        self.emb_size = args_config.dim
        self.channel = args_config.channel
        self.context_hops = args_config.context_hops
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.loss_f = args_config.loss_f
        self.gamma = args_config.gamma
        self.N = args_config.hop_threshold
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
            else torch.device("cpu")

        self.pref_miner = AttentionPreferenceMiner(self.n_users, self.n_items, self.emb_size, 0.1)
        self.fact_embedder = FactSemanticEmbedding(32)

        KGE_model = torch.load(os.path.join("pretrained", "RotatE_last-fm", "trained_model.pkl"), weights_only=False)
        self.entity_embeddings = KGE_model.entity_representations[0](indices=None).detach().to(self.device)
        self.relation_embeddings = KGE_model.relation_representations[0](indices=None).detach().to(self.device)

        self.router = GatingRouter(self.emb_size, self.channel)

        # self.inter_edge: [[user_ids...], [item_ids...]], self.inter_edge_w: [normalized weights of edges, ...]
        # adj_mat: [(user_id, item_id), weight]
        self.inter_edge, self.inter_edge_w = self.convert_sp_mat_to_tensor(adj_mat)

        # self.edge_index: [[head_id, tail_id], ...], self.edge_type: [relation_id + 1, ...]
        # graph: [head_id, tail_id, relation_id + 1] (if inverse relations are considered, relation_id includes inverse relations)
        self.edge_index, self.edge_type = self.get_edges(graph)

        self.init_weight()
        self.all_embed = nn.Parameter(self.all_embed)

        self.init_loss_function()

        self.gcn = GraphConv(emb_size=self.emb_size,
                             n_hops=self.context_hops,
                             n_relations=self.n_relations,
                             mess_dropout_rate=self.mess_dropout_rate,
                             gamma=self.gamma,
                             N=self.N)

    def init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size))

    def init_loss_function(self):
        if self.loss_f == "inner_bpr":
            self.loss = self.create_inner_bpr_loss
        elif self.loss_f == 'contrastive_loss':
            self.loss = self.create_contrastive_loss
        else:
            raise NotImplementedError

    def convert_sp_mat_to_tensor(self, X):  # X: [(user_id, item_id), weight]
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return i.to(self.device), v.to(self.device)

    def get_edges(self, graph):  # graph: [head_id, tail_id, relation_id + 1] (if inverse relation is considered, relation_id includes inverse relations)
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def forward(self, batch=None, gamma=None):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items'].reshape(-1)

        user_emb = self.all_embed[:self.n_users, :]  # [n_users, channel]
        entity_emb = self.all_embed[self.n_users:, :]  # [n_entity, channel]

        """MoE"""
        user_preference = self.pref_miner(self.inter_edge)
        fact_embeddings_tensor = self.fact_embedder(self.edge_index, self.edge_type, self.entity_embeddings, self.relation_embeddings)
        
        # omega = self.mixer(self.user_aggregated, triple_embeddings_tensor)
        omega = self.router(user_preference, fact_embeddings_tensor)

        """node dropout"""
        edge_index, edge_type, omega = relation_aware_edge_sampling(  # node dropout for self.edge_index, self.edge_type, self.omega
            self.edge_index, self.edge_type, omega, self.n_relations, self.node_dropout_rate)
        inter_edge, inter_edge_w = sparse_dropout(  # node dropout for self.inter_edge, self.inter_edge_w
            self.inter_edge, self.inter_edge_w, self.node_dropout_rate)

        """rec task"""
        entity_gcn_emb, user_gcn_emb = self.gcn(user_emb,
                                                entity_emb,
                                                edge_index,
                                                edge_type,
                                                omega,
                                                inter_edge,
                                                inter_edge_w,
                                                mess_dropout=self.mess_dropout,
                                                gamma=gamma
                                                )
        u_e = user_gcn_emb[user]
        pos_e, neg_e = entity_gcn_emb[pos_item], entity_gcn_emb[neg_item]

        loss = self.loss(u_e, pos_e, neg_e)

        return loss
    
    def create_contrastive_loss(self, u_e, pos_e, neg_e):
        batch_size = u_e.shape[0]

        ui_pos_loss = F.relu(1 - F.cosine_similarity(u_e, pos_e, dim=1))

        users_batch = torch.repeat_interleave(u_e, self.num_neg_sample, dim=0)

        ui_neg = F.relu(F.cosine_similarity(users_batch, neg_e, dim=1) - self.margin_ccl)
        ui_neg = ui_neg.view(batch_size, -1)
        x = ui_neg > 0
        ui_neg_loss = torch.sum(ui_neg, dim=-1) / (torch.sum(x, dim=-1) + 1e-5)

        loss = ui_pos_loss + ui_neg_loss

        return loss.mean()

    def create_inner_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        cf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))
        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size

        return cf_loss + emb_loss

    def generate(self, gamma=None):
        user_emb = self.all_embed[:self.n_users, :]
        entity_emb = self.all_embed[self.n_users:, :]
        user_preference = self.pref_miner(self.inter_edge)
        triple_embeddings_tensor = self.fact_embedder(self.edge_index, self.edge_type, self.entity_embeddings, self.relation_embeddings)
        # omega = self.mixer(self.user_aggregated, triple_embeddings_tensor)
        omega = self.router(user_preference, triple_embeddings_tensor)
        return self.gcn(user_emb,
                        entity_emb,
                        self.edge_index,
                        self.edge_type,
                        omega,
                        self.inter_edge,
                        self.inter_edge_w,
                        mess_dropout=False,
                        gamma=gamma)[:2]

    def rating(self, u_g_embeddings, i_g_embeddings):
        if self.loss_f == "inner_bpr":
            return torch.matmul(u_g_embeddings, i_g_embeddings.t())
        elif self.loss_f == 'contrastive_loss':
            u_g_embeddings = F.normalize(u_g_embeddings)
            i_g_embeddings = F.normalize(i_g_embeddings)
            return torch.matmul(u_g_embeddings, i_g_embeddings.t())
