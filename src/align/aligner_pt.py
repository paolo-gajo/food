import torch
from torch.nn import CrossEntropyLoss
from torch.nn.functional import cross_entropy
from transformers import BertPreTrainedModel, BertModel
from icecream import ic

class AlignerLoss(CrossEntropyLoss):
    def __init__(self, len_pred, len_answer, weight = None, size_average=None, ignore_index: int = -100, reduce=None, reduction: str = 'mean', label_smoothing: float = 0) -> None:
        super().__init__(weight, size_average, ignore_index, reduce, reduction, label_smoothing)
        self.len_pred = len_pred.type(torch.float32)
        self.len_answer = len_answer.type(torch.float32)

    def forward(self, input, target):
        diff = abs(self.len_pred - self.len_answer).mean() / self.len_answer.mean()
        ce = cross_entropy(input, target)
        # ce = ce * diff
        return ce

class BertAligner(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config=config, add_pooling_layer=False)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.aligner = torch.nn.Linear(config.hidden_size, 2)
        self.post_init()
    
    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        start_positions = None,
        end_positions = None,
    ):
        bert_outputs = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                )
        H = bert_outputs[0] # torch.Size([16, 107, 768])
        E = self.bert.embeddings(input_ids) # torch.Size([16, 107, 768])

        H_t = H.transpose(1, 2) # torch.Size([16, 768, 107])
        E_t = E.transpose(1, 2) # torch.Size([16, 768, 107])
        indices = torch.where(input_ids == 1729)
        # Create a list of tuples using list comprehension
        tuple_list = [(indices[1][2*i], indices[1][2*i+1]) for i in torch.unique(indices[0])]
        # tuple_tensor = torch.tensor(tuple_list, device='cuda:0')
        source_word_hidden_states_batch_list = []
        for i, tuple in enumerate(tuple_list):
            # source_word_hidden_states = E[i][tuple[0]+1]
            source_word_hidden_states = H[i][tuple[1]-1]
            # source_word_hidden_states = torch.mean(source_word_hidden_states, dim=0)
            source_word_hidden_states_batch_list.append(source_word_hidden_states)

        S = torch.stack(source_word_hidden_states_batch_list).to('cuda') # torch.Size([16, 768])
        # ic(S.shape)
        sim_list = []

        # unnormalized
        # for i in range(S.shape[0]):
        #     sim_list.append(torch.matmul(S[i], H_t[i]))
        # sim = torch.stack(sim_list) # torch.Size([16, 265])
        
        # normalized (cosine similarity)
        for i in range(S.shape[0]):
            # Normalize the vectors
            S_normalized = torch.nn.functional.normalize(S[i], dim=0)
            H_t_normalized = torch.nn.functional.normalize(H_t[i], dim=0)
            # E_t_normalized = torch.nn.functional.normalize(E_t[i], dim=0)

            # Calculate cosine similarity as the dot product of the normalized vectors
            sim_list.append(torch.matmul(S_normalized, H_t_normalized))
            # sim_list.append(torch.matmul(S_normalized, E_t_normalized))

        sim = torch.stack(sim_list)  # torch.Size([16, 265])

        # new_input = []
        # for i in range(S.shape[0]):
        #     sim_list.append(torch.matmul(S[i], H_t[i]))
        # sim = torch.stack(sim_list)

        linear_outputs = self.aligner(H)

        start_logits, end_logits = linear_outputs.split(1, dim=-1)
        
        start_logits = start_logits.squeeze(-1).contiguous()
        start_logits = torch.mul(start_logits, sim)
        
        end_logits = end_logits.squeeze(-1).contiguous()
        end_logits = torch.mul(end_logits, sim)
        # ignored_index = start_logits.size(1)
        # start_positions = start_positions.clamp(0, ignored_index)
        # end_positions = end_positions.clamp(0, ignored_index)
        end_preds = torch.argmax(end_logits, dim=1)
        start_preds = torch.argmax(start_logits, dim=1)
        pred_lengths = end_preds - start_preds
        answer_lengths = end_positions - start_positions

        # loss_fn = AlignerLoss(pred_lengths, answer_lengths)
        loss_fn = CrossEntropyLoss()

        loss_start = loss_fn(start_logits, start_positions)
        loss_end = loss_fn(end_logits, end_positions)

        loss = (loss_start + loss_end)/2

        outputs = {'loss': loss,
                    'start_logits': start_logits,
                    'end_logits': end_logits,
                    'hidden_states': bert_outputs.hidden_states,
                    'attentions': bert_outputs.attentions}
        return outputs
