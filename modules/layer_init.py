
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

VERY_NEG_NUMBER = -100000000000
VERY_SMALL_NUMBER = 1e-10


class TypeLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, device):
        super(TypeLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # self.kb_head_linear = nn.Linear(in_features, out_features)
        self.kb_self_linear = nn.Linear(in_features, out_features)
        # self.kb_tail_linear = nn.Linear(out_features, out_features)
        self.device = device

    # '''
    def forward(self, local_entity, edge_list, rel_features):
        batch_heads, batch_rels, batch_tails, _, _, _  = edge_list

        cat_batch_heads = np.concatenate(batch_heads)
        cat_batch_rels = np.concatenate(batch_rels)
        cat_batch_tails = np.concatenate(batch_tails)
        batch_ids = np.concatenate([np.full(len(batch_heads[i]), i, dtype=np.int) for i in range(len(batch_heads))])
        num_fact = len(cat_batch_heads)
        fact_ids = np.array(range(num_fact), dtype=np.int)

        # head_count = Counter(cat_batch_heads)
        # weight_list = [1.0 / head_count[head] for head in cat_batch_heads]

        # edge_list = cat_batch_heads, cat_batch_rels, cat_batch_tails, batch_ids, fact_ids, weight_list        

        batch_size, max_local_entity = local_entity.size()
        hidden_size = self.in_features
        fact2head = torch.LongTensor([cat_batch_heads, fact_ids]).to(self.device)
        fact2tail = torch.LongTensor([cat_batch_tails, fact_ids]).to(self.device)
        cat_batch_rels = torch.LongTensor(cat_batch_rels).to(self.device)
        batch_ids = torch.LongTensor(batch_ids).to(self.device)
        val_one = torch.ones_like(batch_ids).float().to(self.device)

        fact_rel = torch.index_select(rel_features, dim=0, index=cat_batch_rels)
        fact_val = self.kb_self_linear(fact_rel)

        fact2tail_mat = self._build_sparse_tensor(fact2tail, val_one, (batch_size * max_local_entity, num_fact))
        fact2head_mat = self._build_sparse_tensor(fact2head, val_one, (batch_size * max_local_entity, num_fact))

        f2e_emb = F.relu(torch.sparse.mm(fact2tail_mat, fact_val) + torch.sparse.mm(fact2head_mat, fact_val))
        assert not torch.isnan(f2e_emb).any()
        
        f2e_emb = f2e_emb.view(batch_size, max_local_entity, hidden_size)
        
        return f2e_emb
       
    def _build_sparse_tensor(self, indices, values, size):
        return torch.sparse.FloatTensor(indices, values, size).to(self.device)

class TypeLayer_Extend(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, device, logger):
        super(TypeLayer_Extend, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # self.kb_head_linear = nn.Linear(in_features, out_features)
        self.kb_self_linear = nn.Linear(in_features, out_features)
        # self.kb_tail_linear = nn.Linear(out_features, out_features)
        self.device = device
        # self.w = torch.nn.Parameter(torch.Tensor([1.00]), requires_grad=True)
        self.logger = logger
        
        self.soft_linear = nn.Linear(in_features*3, out_features)
        self.sigmoid = nn.Sigmoid()

    # '''
    def forward(self, f2e_emb, local_entity, edge_list, rel_features, eps):
        # self.logger.info(f"w = {self.w}")

        batch_heads, batch_rels, batch_tails, num_ents, edge_mats, score_mats  = edge_list
        batch_size, max_local_entity = local_entity.size()
        hidden_size = self.in_features
        
        similarity_mats = []
        score_mats = [score_mat.to(self.device) for score_mat in score_mats]
        edge_mats = [edge_mat.to(self.device) for edge_mat in edge_mats]

        for idx in range(len(batch_heads)):
            similarity_mat = -torch.cdist(f2e_emb[idx, :num_ents[idx], :], f2e_emb[idx, :num_ents[idx], :])
            # for i in range(similarity_mat.size()[0]):
            #     similarity_mat[i][i] = VERY_NEG_NUMBER
            similarity_mat = F.softmax(similarity_mat, dim=1)
            similarity_mats.append(similarity_mat)
        probability_mats = [torch.matmul(similarity_mats[idx], score_mats[idx].t()) for idx in range(len(similarity_mats))]
        

        real_batch_heads = []
        real_batch_rels = []
        real_batch_tails = []
        extend_fact = np.array([], dtype=np.int)
        batch_ids = np.array([], dtype=np.int)
        num_fact = 0

        for idx in range(len(batch_heads)):
            batch_head = batch_heads[idx]
            batch_rel = batch_rels[idx]
            batch_tail = batch_tails[idx]

            probability_mat = probability_mats[idx]
            similarity_mat = similarity_mats[idx]
            edge_mat = edge_mats[idx]

            mask = (probability_mat > eps) & (edge_mat < 1)

            row_indices, col_indices = torch.nonzero(mask, as_tuple=True)
            prob_indices = torch.multinomial(similarity_mat[row_indices], 1).squeeze(-1)
            new_rels = edge_mat[prob_indices, col_indices]

            mask = new_rels != 0
            row_indices, new_rels, col_indices = row_indices[mask], new_rels[mask], col_indices[mask]
            row_indices += max_local_entity * idx
            col_indices += max_local_entity * idx
            extend_fact = np.append(extend_fact, [i for i in range(num_fact+len(batch_head), num_fact+len(batch_head)+len(row_indices))])

            batch_head = np.append(batch_head, row_indices.cpu().numpy().astype(int))
            batch_rel = np.append(batch_rel, new_rels.cpu().numpy().astype(int))
            batch_tail = np.append(batch_tail, col_indices.cpu().numpy().astype(int))
            num_fact += len(batch_head)
            
            real_batch_heads.append(batch_head)
            real_batch_rels.append(batch_rel)
            real_batch_tails.append(batch_tail)     
            batch_ids = np.append(batch_ids, [idx for _ in range(len(batch_head))])

        edge_list = real_batch_heads, real_batch_rels, real_batch_tails, num_ents, edge_mats, score_mats 

        real_batch_heads = np.concatenate(real_batch_heads)
        real_batch_rels = np.concatenate(real_batch_rels)
        real_batch_tails = np.concatenate(real_batch_tails)

        fact_ids = np.array([i for i in range(len(real_batch_heads))])

        fact2head = torch.LongTensor([real_batch_heads, fact_ids]).to(self.device)
        fact2tail = torch.LongTensor([real_batch_tails, fact_ids]).to(self.device)
        not_real_rels = torch.LongTensor(real_batch_rels).to(self.device)
        val_one = torch.ones_like(not_real_rels).float().to(self.device)
        
        num_fact = len(real_batch_heads)
        fact_rel = torch.index_select(rel_features, dim=0, index=not_real_rels)
        # fact_rel[extend_fact, :] += 0.03 * torch.normal(0, 0.2, (len(extend_fact), fact_rel.shape[1])).to(self.device)
        # fact_rel[extend_fact, :] *= self.w

        fact_val = self.kb_self_linear(fact_rel)
        fact2tail_mat = self._build_sparse_tensor(fact2tail, val_one, (batch_size * max_local_entity, num_fact))
        fact2head_mat = self._build_sparse_tensor(fact2head, val_one, (batch_size * max_local_entity, num_fact))
        
        '''
        f2e_emb_cat = f2e_emb.view(-1, f2e_emb.shape[2])
        try:
            try_emb = torch.cat((f2e_emb_cat[real_batch_heads[extend_fact], :],  rel_features[real_batch_rels[extend_fact], :], f2e_emb_cat[real_batch_tails[extend_fact], :]), dim=1)
        except:
            extend_fact = extend_fact.astype(int)
            try_emb = torch.cat((f2e_emb_cat[real_batch_heads[extend_fact], :],  rel_features[real_batch_rels[extend_fact], :], f2e_emb_cat[real_batch_tails[extend_fact], :]), dim=1)
        
        A = self.soft_linear(try_emb)
        B = self.sigmoid(A)
        fact_val[extend_fact, :] = torch.mul(fact_val[extend_fact, :], B)        
        '''

        
        real_emb = F.relu(torch.sparse.mm(fact2tail_mat, fact_val) + torch.sparse.mm(fact2head_mat, fact_val))

        real_emb = real_emb.view(batch_size, max_local_entity, hidden_size)

        return real_emb, edge_list

    def _build_sparse_tensor(self, indices, values, size):
        return torch.sparse.FloatTensor(indices, values, size).to(self.device)
    
