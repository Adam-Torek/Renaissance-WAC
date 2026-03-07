from timm.layers import classifier
from timm.utils import summary
from torch.nn import attention
from transformers import ElectraConfig, LxmertConfig, AutoConfig, ViTConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.models import vit

class WACConfig(ElectraConfig):

    model_type = "bert_wac"

    def __init__(self, 
                 wac_embedding_size=None, 
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
                 layer_norm_eps=1e-8,
                 summary_type="first",
                 summary_use_proj=True,
                 summary_activation="gelu",
                 summary_last_dropout=0.1,
                 pad_token_id=0,
                 position_embedding_type="absolute",
                 use_cache=True,
                 classifier_dropout=None,
                 ignore_text_embeddings_epochs=2,
                 **kwargs) -> None:
        
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
                         classifier_dropout=classifier_dropout,
                         **kwargs)
        
        self.wac_embedding_size = wac_embedding_size
        self.wac_distribution_matrix = wac_distribution_matrix
        self.ignore_text_embeddings_epochs = ignore_text_embeddings_epochs
    
class TwoTowerConfig(PretrainedConfig):

    model_type = "two_tower"

    def __init__(self, 
                 config=None, 
                 wac_embedding_size=None,
                 **kwargs):

        if config is not None:
            self.random_init_text_encoder = config['random_init_text_encoder']
            self.random_init_vision_encoder = config['random_init_vision_encoder']
            self.use_mask_token = config["loss_names"]["mim"] > 0

            if config['use_text_encoder']:
                self.text_encoder_path = config["text_encoder"]
                self.freeze_text_encoder = config["freeze_text_encoder"]

                if config['text_encoder_manual_configuration']:
                    text_encoder_kwargs = {
                        'hidden_size' : config["text_encoder_hidden_size"],
                        'num_hidden_layers' : config["text_encoder_num_layers"],
                        'num_attention_heads' : config["text_encoder_num_heads"],
                        'intermediate_size' : config["text_encoder_hidden_size"] * config["text_encoder_mlp_ratio"],
                        'hidden_dropout_prob' : config["text_encoder_drop_rate"],
                        'attention_probs_dropout_prob' : config["text_encoder_drop_rate"],
                        'layer_norm_eps': config['text_encoder_norm_eps'],
                    }
                    if 'electra' in config['text_encoder']:
                        text_encoder_kwargs['embedding_size'] = config['text_encoder_embedding_size']
                    if config['model_type'] == 'two-tower-wac':
                        text_encoder_kwargs['wac_embedding_size'] = wac_embedding_size
                        text_encoder_kwargs['wac_distribution_matrix'] = config['wac_distribution_matrix']
                        text_encoder_kwargs['ignore_text_embeddings_epochs'] = config['ignore_text_embeddings_epochs']
                        self.text_config = WACConfig(**text_encoder_kwargs)
                    else:
                        self.text_config = AutoConfig.from_pretrained(config['text_encoder'], **text_encoder_kwargs)
                else:
                    if config['model_type'] == 'two-tower-wac':
                        wac_encoder_kwargs = {
                            'wac_embedding_size': wac_embedding_size,
                            'wac_distribution_matrix': config['wac_distribution_matrix'],
                            'ignore_text_embeddings_epochs': config['ignore_text_embeddings_epochs'],
                            }
                        
                        self.text_config = WACConfig(**wac_encoder_kwargs)
                    else:
                        self.text_config = AutoConfig.from_pretrained(config['text_encoder'])
            else:
                self.text_encoder_path = None
                self.text_config = None

            if config['use_image_encoder']:
                self.image_encoder_path = config["image_encoder"]
                self.freeze_image_encoder = config["freeze_image_encoder"]

                if config['image_encoder_manual_configuration']:
                    image_encoder_kwargs = {
                        'hidden_size' : config["image_encoder_hidden_size"],
                        'num_hidden_layers' : config["image_encoder_num_layers"],
                        'num_attention_heads' : config["image_encoder_num_heads"],
                        'intermediate_size' : config["image_encoder_hidden_size"] * config["image_encoder_mlp_ratio"],
                        'hidden_dropout_prob' : config["image_encoder_drop_rate"],
                        'attention_probs_dropout_prob' : config["image_encoder_drop_rate"],
                        'layer_norm_eps': config['layer_norm_eps'],
                    }
                    self.image_config = AutoConfig.from_pretrained(config["image_encoder"], **image_encoder_kwargs)
                else:
                    self.image_config = AutoConfig.from_pretrained(config["image_encoder"])
            else:
                self.image_encoder_path = None
                self.image_config = None

            if config["use_image_encoder"] and config["use_text_encoder"]:
                self.cross_layer_hidden_size = config["cross_layer_hidden_size"]
                self.freeze_cross_modal_layers = config["freeze_cross_modal_layers"]

                self.fusion_config = LxmertConfig(
                    vocab_size=config["vocab_size"],
                    hidden_size=config["cross_layer_hidden_size"],
                    num_attention_heads=config["num_cross_layer_heads"],
                    intermediate_size=config["cross_layer_hidden_size"] * config["cross_layer_mlp_ratio"],
                    max_position_embeddings=config["max_text_len"],
                    hidden_dropout_prob=config["cross_layer_drop_rate"],
                    attention_probs_dropout_prob=config["cross_layer_drop_rate"],
                    layer_norm_eps=config['cross_encoder_norm_eps'],
                )  
            else:
                self.fusion_config = None      

        else:
            self.use_mask_token = False
            self.text_config = ElectraConfig()
            self.image_config = ViTConfig()
            self.fusion_config = LxmertConfig(vocab_size=self.text_config.vocab_size,
                                              hidden_size=self.text_config.hidden_size,
                                              num_attention_heads=self.text_config.num_attention_heads,
                                              intermediate_size=self.text_config.intermediate_size,
                                              max_position_embeddings=self.text_config.max_position_embeddings,
                                              hidden_dropout_prob=self.text_config.hidden_dropout_prob,
                                              attention_probs_dropout_prob=self.text_config.attention_probs_dropout_prob)

        super().__init__(**kwargs)

class OneTowerConfig(PretrainedConfig):

    model_type = "one_tower"

    def __init__(self, 
                 config=None, 
                 image_size=224, 
                 max_text_len=50, 
                 **kwargs):

        self.image_size = image_size
        self.max_text_len = max_text_len

        if config is not None:
            self.encoder_path = config['encoder']
            self.pooler_type = config['pooler_type']

            if config['random_init_encoder']:
                # Manually Configure Encoder Dimensions
                if config['encoder_manual_configuration']:
                    encoder_kwargs = {
                        'vocab_size' : config["vocab_size"],
                        'hidden_size' : config["hidden_size"],
                        'num_hidden_layers' : config["num_layers"],
                        'num_attention_heads' : config["num_heads"],
                        'intermediate_size' : config["hidden_size"] * config["mlp_ratio"],
                        'max_position_embeddings' : config["max_text_len"],
                        'hidden_dropout_prob' : config["drop_rate"],
                        'attention_probs_dropout_prob' : config["drop_rate"],
                    }
                    self.one_tower_config = AutoConfig.from_pretrained(config['encoder'], **encoder_kwargs)
                # Use Default Encoder Dimensions with Random Weights
                elif not config['encoder_manual_configuration']:
                    self.one_tower_config = AutoConfig.from_pretrained(config['encoder'])
                
            else:
                self.one_tower_config = None

            self.image_config = ViTConfig(
            image_size=image_size,
            patch_size=config['patch_size'],
            hidden_size=config["text_encoder_embedding_size"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )
        
            self.text_config = ElectraConfig(
                vocab_size=config["vocab_size"],
                hidden_size=config["text_encoder_hidden_size"],
                embedding_size=config["text_encoder_embedding_size"],
                max_position_embeddings=max_text_len,
                hidden_dropout_prob=config["drop_rate"],
                attention_probs_dropout_prob=config["drop_rate"],
            )

        else:
            self.one_tower_config = ElectraConfig()
            self.text_config = self.one_tower_config
            self.image_config = ViTConfig(image_size=image_size,
                                          patch_size=16,
                                          hidden_size=self.text_config.embedding_size,
                                          hidden_dropout_prob=self.text_config.hidden_dropout_prob,
                                          attention_probs_dropout_prob=self.text_config.attention_probs_dropout_prob,)

        super().__init__(**kwargs)