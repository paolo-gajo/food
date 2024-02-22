import torch
from torch.nn import CrossEntropyLoss
from torch.nn.functional import cross_entropy
from transformers import BertPreTrainedModel, BertModel

class AlignerLoss(CrossEntropyLoss):
    def __init__(self, len_pred, len_answer, weight = None, size_average=None, ignore_index: int = -100, reduce=None, reduction: str = 'mean', label_smoothing: float = 0) -> None:
        super().__init__(weight, size_average, ignore_index, reduce, reduction, label_smoothing)
        self.len_pred = len_pred.type(torch.float32)
        self.len_answer = len_answer.type(torch.float32)
    def forward(self, input, target):
        diff = (abs(self.len_pred - self.len_answer)**2).mean().clamp(0,1)
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
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        linear_outputs = self.aligner(bert_outputs[0])
        start_logits, end_logits = linear_outputs.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
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
