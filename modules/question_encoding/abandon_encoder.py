from transformers import RobertaTokenizer, RobertaModel
import torch
import torch.nn as nn
import numpy as np

class FakeEncoder(nn.Module):
    def __init__(self, in_features, out_features, device):
        super(FakeEncoder, self).__init__()
        self.device = device
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.model = RobertaModel.from_pretrained("roberta-base")
        self.lower = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
        self.to(self.device)

    '''
    def forward(self, inputs, local_entity_emb):
        inputs = np.where(inputs==None, 'None', inputs).tolist()
        embeddings = []

        for input in inputs:
            input = self.tokenizer(input, return_tensors="pt", padding=True, truncation=True)
            input = input.to(self.device)
            with torch.no_grad():
                output = self.model(**input)
            output = output.last_hidden_state.mean(dim=1)
            embedding = self.lower(output)
            embeddings.append(embedding)
        
        embeddings = torch.stack(embeddings, dim=0)
        embeddings = torch.cat((local_entity_emb, embeddings), dim=2)
        embeddings = self.efls(embeddings)

        return embeddings    
    '''
    
    def forward(self, fake_inputs):
        # '''
        inputs = fake_inputs.tolist()
        embeddings = []

        for input in inputs:
            # mask = []
            # for i in range(len(input)):
            #     if input[i] == ' ': mask.append(i)
            
            # mask = [i == ' ' for i in input]
            input = self.tokenizer(input, return_tensors="pt", padding=True, truncation=True)
            input = input.to(self.device)
            
            with torch.no_grad():
                output = self.model(**input)
            
            embedding = output.last_hidden_state.mean(dim=1)
            # embedding[mask, :] = torch.zeros(768, dtype=torch.float).to(self.device)
            embeddings.append(embedding)
        
        embeddings = torch.stack(embeddings, dim=0)
        embeddings = self.lower(embeddings)
        
        '''
        batch_size = len(fake_inputs)
        # inputs = np.where(inputs==None, ' ', inputs)
        
        inputs = np.concatenate(fake_inputs)
        # mask = inputs == ' '
        inputs = inputs.tolist()
        
        inputs = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            output = self.model(**inputs)
            
        # output = output.last_hidden_state.mean(dim=1).cpu().numpy()
        # # output[mask, :] = np.zeros((1, 768), np.float32)
        # output = torch.from_numpy(output).to(self.device)
        
        output = output.last_hidden_state.mean(dim=1)
        output = output.view(batch_size, -1, output.shape[1])
        
        embeddings = self.lower(output)
        
        # '''
        
        return embeddings

# class RealEncoder(nn.Module):
#     def __init__(self, in_features, out_features, device):
#         super(RealEncoder, self).__init__()
#         self.device = device
#         self.efls = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
#         self.to(self.device)

#     def forward(self, inputs, local_entity_emb):
#         inputs = np.where(inputs==None, 'None', inputs).tolist()
#         embeddings = []

#         for input in inputs:
#             input = self.tokenizer(input, return_tensors="pt", padding=True, truncation=True)
#             input = input.to(self.device)
#             with torch.no_grad():
#                 output = self.model(**input)
#             output = output.last_hidden_state.mean(dim=1)
#             embedding = self.lower(output)
#             embeddings.append(embedding)
        
#         embeddings = torch.stack(embeddings, dim=0)
#         embeddings = torch.cat((local_entity_emb, embeddings), dim=2)
#         embeddings = self.efls(embeddings)

#         return embeddings
        