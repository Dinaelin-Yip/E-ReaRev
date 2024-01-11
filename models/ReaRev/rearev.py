import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from thop import profile

from models.base_model import BaseModel
from modules.kg_reasoning.reasongnn import ReasonGNNLayer
from modules.question_encoding.lstm_encoder import LSTMInstruction
from modules.question_encoding.bert_encoder import BERTInstruction
from modules.question_encoding.abandon_encoder import FakeEncoder
from modules.layer_init import TypeLayer, TypeLayer_Extend
from modules.query_update import AttnEncoder, Fusion, QueryReform

from collections import Counter

VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000



class ReaRev(BaseModel):
    def __init__(self, args, logger, num_entity, num_relation, num_word):
        """
        Init ReaRev model.
        """
        super(ReaRev, self).__init__(args, num_entity, num_relation, num_word)
        #self.embedding_def()
        #self.share_module_def()
        
        self.loss_type =  args['loss_type']
        self.num_iter = args['num_iter']
        self.num_ins = args['num_ins']
        self.num_gnn = args['num_gnn']
        self.alg = args['alg']
        assert self.alg == 'bfs'
        self.lm = args['lm']

        # edge_extension
        self.edge_extension = args['edge_extension']
        self.eps = args['eps_fake']
        self.mean_extension = args['mean_extension']
        
        self.private_module_def(args, num_entity, num_relation)

        self.to(self.device)
        self.lin = nn.Linear(3*self.entity_dim, self.entity_dim)

        self.fusion = Fusion(self.entity_dim)
        self.reforms = []
        for i in range(self.num_ins):
            self.add_module('reform' + str(i), QueryReform(self.entity_dim))
        
        self.logger = logger
        self.layers(args)

        


    def layers(self, args):
        # initialize entity embedding
        word_dim = self.word_dim
        kg_dim = self.kg_dim
        entity_dim = self.entity_dim

        #self.lstm_dropout = args['lstm_dropout']
        self.linear_dropout = args['linear_dropout']
        
        self.entity_linear = nn.Linear(in_features=self.ent_dim, out_features=entity_dim)
        # self.relation_linear = nn.Linear(in_features=self.rel_dim, out_features=entity_dim)
        # self.relation_linear_inv = nn.Linear(in_features=self.rel_dim, out_features=entity_dim)
        #self.relation_linear = nn.Linear(in_features=self.rel_dim, out_features=entity_dim)

        # dropout
        #self.lstm_drop = nn.Dropout(p=self.lstm_dropout)
        self.linear_drop = nn.Dropout(p=self.linear_dropout)

        self.type_layer = TypeLayer(in_features=entity_dim, out_features=entity_dim, device=self.device)
        if self.edge_extension:
            self.extend_layer = TypeLayer_Extend(in_features=entity_dim, out_features=entity_dim, device=self.device, logger=self.logger)
            self.extend_layer.kb_self_linear = self.type_layer.kb_self_linear  # share one relation layer
        if self.mean_extension:
            self.fake_encoder = FakeEncoder(in_features=768, out_features=entity_dim, device=self.device)

        self.self_att_r = AttnEncoder(self.entity_dim)
        #self.self_att_r_inv = AttnEncoder(self.entity_dim)
        self.kld_loss = nn.KLDivLoss(reduction='none')
        self.bce_loss_logits = nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss()

        self.lower = nn.Linear(in_features=entity_dim*2, out_features=entity_dim, bias=True).to(self.device)
   
    def get_rel_feature(self):
        """
        Encode relation tokens to vectors.
        """
        if self.rel_texts is None:
            rel_features = self.relation_embedding.weight
            rel_features_inv = self.relation_embedding_inv.weight
            rel_features = self.relation_linear(rel_features)
            rel_features_inv = self.relation_linear(rel_features_inv)
        else:
            
            rel_features = self.instruction.question_emb(self.rel_features)
            rel_features_inv = self.instruction.question_emb(self.rel_features_inv)
            
            rel_features = self.self_att_r(rel_features,  (self.rel_texts != self.instruction.pad_val).float())
            rel_features_inv = self.self_att_r(rel_features_inv,  (self.rel_texts != self.instruction.pad_val).float())
            if self.lm == 'lstm':
                rel_features = self.self_att_r(rel_features, (self.rel_texts != self.num_relation+1).float())
                rel_features_inv = self.self_att_r(rel_features_inv, (self.rel_texts_inv != self.num_relation+1).float())

        return rel_features, rel_features_inv


    def private_module_def(self, args, num_entity, num_relation):
        """
        Building modules: LM encoder, GNN, etc.
        """
        # initialize entity embedding
        word_dim = self.word_dim
        kg_dim = self.kg_dim
        entity_dim = self.entity_dim
        self.reasoning = ReasonGNNLayer(args, num_entity, num_relation, entity_dim, self.alg)
        if args['lm'] == 'lstm':
            self.instruction = LSTMInstruction(args, self.word_embedding, self.num_word)
            self.relation_linear = nn.Linear(in_features=entity_dim, out_features=entity_dim)
        else:
            self.instruction = BERTInstruction(args, self.word_embedding, self.num_word, args['lm'])
            #self.relation_linear = nn.Linear(in_features=self.instruction.word_dim, out_features=entity_dim)
        # self.relation_linear = nn.Linear(in_features=entity_dim, out_features=entity_dim)
        # self.relation_linear_inv = nn.Linear(in_features=entity_dim, out_features=entity_dim)

    def build_new_kb(self, edge_list):
        batch_heads, batch_rels, batch_tails, _, _, _  = edge_list
        cat_batch_heads = np.concatenate(batch_heads)
        cat_batch_rels = np.concatenate(batch_rels)
        cat_batch_tails = np.concatenate(batch_tails)

        batch_ids = np.concatenate([np.full(len(batch_heads[i]), i, dtype=int) for i in range(len(batch_heads))])
        num_fact = len(cat_batch_heads)
        fact_ids = np.array(range(num_fact), dtype=int)

        head_count = Counter(cat_batch_heads)
        weight_list = [1.0 / head_count[head] for head in cat_batch_heads]

        edge_list = cat_batch_heads, cat_batch_rels, cat_batch_tails, batch_ids, fact_ids, weight_list

        return edge_list

    def init_reason(self, curr_dist, local_entity, kb_adj_mat, q_input, query_entities, actual_entities):
        """
        Initializing Reasoning
        """
        # batch_size = local_entity.size(0)
        self.local_entity = local_entity
        self.instruction_list, self.attn_list = self.instruction(q_input)
        rel_features, rel_features_inv  = self.get_rel_feature()
        
        init_emb = self.type_layer(local_entity, kb_adj_mat, rel_features)
        if self.edge_extension:
            if self.mean_extension:
                fake_emb = self.fake_encoder(actual_entities)
                generated_emb = self.lower(torch.cat((init_emb, fake_emb), dim=2))
                generated_emb, kb_adj_mat = self.extend_layer(generated_emb, local_entity, kb_adj_mat, rel_features, self.eps)
                self.init_entity_emb = self.lower(torch.cat((generated_emb, fake_emb), dim=2))
            else:
                self.init_entity_emb, kb_adj_mat = self.extend_layer(init_emb, local_entity, kb_adj_mat, rel_features, self.eps)
        elif self.mean_extension:
            fake_emb = self.fake_encoder(actual_entities)
            self.init_entity_emb = self.lower(torch.cat((init_emb, fake_emb), dim=2))
        else:
            self.init_entity_emb = init_emb
        kb_adj_mat = self.build_new_kb(kb_adj_mat)
        self.curr_dist = curr_dist
        self.dist_history = []
        self.action_probs = []
        self.seed_entities = curr_dist
        
        self.reasoning.init_reason( 
                                   local_entity=local_entity,
                                   kb_adj_mat=kb_adj_mat,
                                   local_entity_emb=self.init_entity_emb,
                                   rel_features=rel_features,
                                   rel_features_inv=rel_features_inv,
                                   query_entities=query_entities)


    def calc_loss_label(self, curr_dist, teacher_dist, label_valid):
        tp_loss = self.get_loss(pred_dist=curr_dist, answer_dist=teacher_dist, reduction='none')
        tp_loss = tp_loss * label_valid
        cur_loss = torch.sum(tp_loss) / curr_dist.size(0)
        return cur_loss

    
    def forward(self, batch, training=False):
        """
        Forward function: creates instructions and performs GNN reasoning.
        """

        # local_entity, query_entities, kb_adj_mat, query_text, seed_dist, answer_dist = batch
        local_entity, query_entities, kb_adj_mat, query_text, seed_dist, actual_entities, answer_dist = batch
        local_entity = torch.from_numpy(local_entity).type('torch.LongTensor').to(self.device)
        # local_entity_mask = (local_entity != self.num_entity).float()
        query_entities = torch.from_numpy(query_entities).type('torch.FloatTensor').to(self.device)
        answer_dist = torch.from_numpy(answer_dist).type('torch.FloatTensor').to(self.device)
        seed_dist = torch.from_numpy(seed_dist).type('torch.FloatTensor').to(self.device)
        current_dist = Variable(seed_dist, requires_grad=True)

        q_input= torch.from_numpy(query_text).type('torch.LongTensor').to(self.device)
        #query_text2 = torch.from_numpy(query_text2).type('torch.LongTensor').to(self.device)
        if self.lm != 'lstm':
            pad_val = self.instruction.pad_val #tokenizer.convert_tokens_to_ids(self.instruction.tokenizer.pad_token)
            query_mask = (q_input != pad_val).float()
            
        else:
            query_mask = (q_input != self.num_word).float()

        
        """
        Instruction generations
        """
        self.init_reason(curr_dist=current_dist, local_entity=local_entity,
                         kb_adj_mat=kb_adj_mat, q_input=q_input, query_entities=query_entities, actual_entities=actual_entities)
        self.instruction.init_reason(q_input)
        for i in range(self.num_ins):
            relational_ins, attn_weight = self.instruction.get_instruction(self.instruction.relational_ins, step=i) 
            self.instruction.instructions.append(relational_ins.unsqueeze(1))
            self.instruction.relational_ins = relational_ins
        #relation_ins = torch.cat(self.instruction.instructions, dim=1)
        #query_emb = None
        self.dist_history.append(self.curr_dist)


        """
        BFS + GNN reasoning
        """

        for t in range(self.num_iter):
            relation_ins = torch.cat(self.instruction.instructions, dim=1)
            self.curr_dist = current_dist            
            for j in range(self.num_gnn):
                self.curr_dist, global_rep = self.reasoning(self.curr_dist, relation_ins, step=j)
            self.dist_history.append(self.curr_dist)
            qs = []

            """
            Instruction Updates
            """
            if t != self.num_iter-1:
                for j in range(self.num_ins):
                    reform = getattr(self, 'reform' + str(j))
                    q = reform(self.instruction.instructions[j].squeeze(1), global_rep, query_entities, local_entity)
                    qs.append(q.unsqueeze(1))
                    self.instruction.instructions[j] = q.unsqueeze(1)
        
        
        """
        Answer Predictions
        """
        
        pred_dist = self.dist_history[-1]
        answer_number = torch.sum(answer_dist, dim=1, keepdim=True)
        case_valid = (answer_number > 0).float()
        # filter no answer training case
        # loss = 0
        # for pred_dist in self.dist_history:
        loss = self.calc_loss_label(curr_dist=pred_dist, teacher_dist=answer_dist, label_valid=case_valid)

        
        pred_dist = self.dist_history[-1]
        pred = torch.max(pred_dist, dim=1)[1]
        if training:
            h1, f1 = self.get_eval_metric(pred_dist, answer_dist)
            tp_list = [h1.tolist(), f1.tolist()]
        else:
            tp_list = None
        return loss, pred, pred_dist, tp_list

    