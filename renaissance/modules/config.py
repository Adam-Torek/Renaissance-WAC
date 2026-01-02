from timm.layers import classifier
from timm.utils import summary
from torch.nn import attention
from transformers import ElectraConfig

class WACConfig(ElectraConfig):

    def __init__(self, 
                 wac_embedding_size=-1, 
                 wac_distribution_matrix=None,
                 vocab_size=30522,
                 embedding_size=128,
                 hidden_size=256,
                 num_hidden_layers=12,
                 num_attention_heads=4,
                 intermediate_size=1024,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 summary_type="first",
                 summary_use_proj=True,
                 summary_activation="gelu",
                 summary_last_dropout=0.1,
                 pad_token_id=0,
                 position_embedding_type="absolute",
                 use_cache=True,
                 classifier_dropout=None,) -> None:
        
        super().__init__(vocab_size=vocab_size,
                         embedding_size=embedding_size,
                         hidden_size=hidden_size,
                         num_hidden_layers=num_hidden_layers,
                         num_attention_heads=num_attention_heads,
                         intermediate_size=intermediate_size,
                         hidden_act=hidden_act,
                         hidden_dropout_prob=hidden_dropout_prob,
                         attention_probs_dropout_prob=attention_probs_dropout_prob,
                         max_position_embeddings=max_position_embeddings,
                         type_vocab_size=type_vocab_size,
                         initializer_range=initializer_range,
                         layer_norm_eps=layer_norm_eps,
                         summary_type=summary_type,
                         summary_use_proj=summary_use_proj,
                         summary_activation=summary_activation,
                         summary_last_dropout=summary_last_dropout,
                         pad_token_id=pad_token_id,
                         position_embedding_type=position_embedding_type,
                         use_cache=use_cache,
                         classifier_dropout=classifier_dropout)
        
        self.wac_embedding_size = wac_embedding_size
        self.wac_distribution_matrix = wac_distribution_matrix
    