import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.models import FairseqEncoder, FairseqIncrementalDecoder, FairseqEncoderDecoderModel, register_model
from fairseq.utils import get_incremental_state, set_incremental_state

from fairseq.data.dictionary import Dictionary


def new_zeros(module: nn.Module, size, dtype=None, requires_grad=False):
    return next(module.parameters()).new_zeros(size, dtype=dtype, requires_grad=requires_grad)


class NICEncoder(FairseqEncoder):
    def __init__(self, args, dictionary: Dictionary, feat_dim, word_emb_dim):
        super().__init__(dictionary)
        self.fc = nn.Linear(in_features=feat_dim, out_features=word_emb_dim)

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        """
        :param src_tokens: {'feat_fc': (batch_size, feat_dim)}
        :param src_lengths:
        :param kwargs:
        :return:
        """
        feat_fc = kwargs['feat_fc']
        return {'feat_fc': self.fc(feat_fc)}

    def reorder_encoder_out(self, encoder_out, new_order):
        feat_fc = encoder_out['feat_fc']
        return {'feat_fc': torch.index_select(feat_fc, dim=0, index=new_order)}


class NICDecoder(FairseqIncrementalDecoder):
    def __init__(self, args, dictionary: Dictionary, word_emb_dim, hidden_dim):
        super().__init__(dictionary)
        self.word_emb = nn.Embedding(num_embeddings=len(self.dictionary), embedding_dim=word_emb_dim,
                                     padding_idx=self.dictionary.pad_index)
        self.rnn = nn.LSTMCell(input_size=word_emb_dim, hidden_size=hidden_dim)
        self.output_fc = nn.Linear(in_features=hidden_dim, out_features=len(self.dictionary))

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None, **kwargs):
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]

        feat_fc = encoder_out['feat_fc']
        batch_size, max_len = prev_output_tokens.shape

        step, rnn_state = (get_incremental_state(self, incremental_state, key=_key) for _key in ('step', 'rnn_state'))
        if step is None:
            step = new_zeros(self, size=(batch_size,), dtype=torch.int64)
        if rnn_state is None:   # first step
            rnn_state = self.rnn.forward(input=feat_fc, hx=None)

        # print('step', step.detach().cpu().numpy(), 'prev_output_tokens', prev_output_tokens.detach().cpu().numpy().reshape(-1))

        prev_output_emb = self.word_emb(prev_output_tokens)
        all_logits = []
        for i in range(max_len):
            word_input = prev_output_emb[:, i]
            rnn_state = self.rnn.forward(word_input, rnn_state)
            (h, c) = rnn_state
            step = step + 1
            all_logits.append(self.output_fc(h))

        for _key, _value in [('step', step), ('rnn_state', rnn_state)]:
            set_incremental_state(self, incremental_state, key=_key, value=_value)

        all_logits = torch.stack(all_logits, dim=1)     # (batch_size, max_len, vocab_size)
        return all_logits, None

    def reorder_incremental_state(self, incremental_state, new_order):
        super(NICDecoder, self).reorder_incremental_state(incremental_state, new_order)
        step, rnn_state = (get_incremental_state(self, incremental_state, key=_key) for _key in ('step', 'rnn_state'))
        step = torch.index_select(step, dim=0, index=new_order)
        rnn_state = (torch.index_select(rnn_state[0], dim=0, index=new_order),
                     torch.index_select(rnn_state[1], dim=0, index=new_order))
        for _key, _value in [('step', step), ('rnn_state', rnn_state)]:
            set_incremental_state(self, incremental_state, key=_key, value=_value)


@register_model('nic')
class NICModel(FairseqEncoderDecoderModel):
    @staticmethod
    def add_args(parser):
        parser.add_argument('-feat_dim', type=int, default=2048)
        parser.add_argument('-word_emb_dim', type=int, default=300)
        parser.add_argument('-hidden_dim', type=int, default=512)

    @staticmethod
    def build_model(args, task, **kwargs):
        dictionary = kwargs.get('dictionary', None)
        if dictionary is None:
            dictionary = task.dictionary

        feat_dim, word_emb_dim, hidden_dim = args.feat_dim, args.word_emb_dim, args.hidden_dim
        encoder = NICEncoder(args, dictionary=dictionary, word_emb_dim=word_emb_dim, feat_dim=feat_dim)
        decoder = NICDecoder(args, dictionary=dictionary, word_emb_dim=word_emb_dim, hidden_dim=hidden_dim)
        model = NICModel(encoder, decoder)
        print(model)
        return model

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        return decoder_out

    def forward_decoder(self, prev_output_tokens, **kwargs):
        return self.decoder(prev_output_tokens, **kwargs)

