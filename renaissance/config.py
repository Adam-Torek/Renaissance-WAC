from sacred import Experiment

ex = Experiment("renaissance")

def _loss_names(d):
    ret = {
        "itm": 0,
        "mim": 0,
        "mlm": 0,
        "mpp": 0,
        "vqa": 0,
        "vcr": 0,
        "vcr_qar": 0,
        "nlvr2": 0,
        "irtr": 0,
        "contras": 0,
        "snli": 0,
        "ref": 0,
        "ref_bbox": 0,
        "mrpc" : 0,
        "rte" : 0,
        'wnli' : 0,
        'sst2' : 0,
        'stsb': 0,
        'qqp' : 0,
        'qnli' : 0,
        'mnli' : 0,
        'cola' : 0,
        'cifar10' : 0
    }
    ret.update(d)
    return ret

# ===================== Default Settings ===================== #
@ex.config
def config():
    exp_name = "renaissance"
    seed = 0
    datasets = ["coco", "vg", "babylm"] # Supports ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 256  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.
    per_gpu_batchsize = 0  # you should define this manually with per_gpu_batch_size=#
    eval_batch_size = 32
    
    # Path to .ckpt file for fine-tuning or testing
    load_path = ""
    # Path to .ckpt file for resuming training from previous checkpoint
    resume_from = None

    # Model Type Setting
    model_type = "two-tower" # Supports ['one-tower', 'two-tower]
    
    #### One Tower Settings ####
    # one-tower settings will be ignored if unless model_type = "one-tower"
    # Text Setting
    encoder = "google/electra-small-discriminator"
    pooler_type = 'double' # Supports ['single', 'double']
    tokenizer = "bert-base-uncased"

    # Transformer Settings

    # Load pretrained model from HuggingFace
    complete_encoder_path = None

    # Train encoder model from scratch
    random_init_encoder = False
    ## Manual Configuration
    encoder_manual_configuration = False
    hidden_size = 192
    num_heads = 4
    num_layers = 12
    mlp_ratio = 4
    drop_rate = 0.1
    embedding_size = 96

    #### Two Tower Settings ####
    ### Image Encoder settings
    image_encoder = "facebook/deit-tiny-patch16-224"
    ## Train encoder model from scratch
    random_init_vision_encoder = False
    ## Manual Configure Image Encoder
    image_encoder_manual_configuration = False
    ## Manual Configuration
    image_encoder_hidden_size = 192
    image_encoder_num_heads = 4
    image_encoder_num_layers = 12
    image_encoder_mlp_ratio = 4
    image_encoder_drop_rate = 0.1
    image_encoder_embedding_size = 128
    image_encoder_norm_eps = 1e-8
    image_size = 224
    original_image_size = 224 # Image size model is pretrained with, used in fine-tuning and testing
    patch_size = 16
    encoder_stride = 16
    num_channels = 3
    image_mask_prob = 0.75
    image_only = False
    use_text_encoder = True
    use_image_encoder = True
    # Image Transform Keys
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]

    
    # Text Setting
    text_encoder = "google/electra-small-discriminator"
    # Train Text Encoder from Sratch if True
    random_init_text_encoder = False
    # Manual Text Settings - Ignored unless random_init_text_encoder = True
    text_encoder_manual_configuration = False
    text_encoder_hidden_size = 192
    text_encoder_num_heads = 4
    text_encoder_num_layers = 12
    text_encoder_mlp_ratio = 4
    text_encoder_drop_rate = 0.1
    text_encoder_embedding_size = 64
    text_encoder_norm_eps = 1e-8
    max_text_len = 40
    vocab_size = 30522
    ignore_text_embeddings_epochs = 2

    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 6
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1
    cross_encoder_norm_eps = 1e-8
    
    # Freeze Module Parameter Settings
    freeze_image_encoder = False
    freeze_text_encoder = False
    freeze_cross_modal_layers = False   
    
    # Pretraining Settings
    # Masked Language Mmodeling
    whole_word_masking = False # note that whole_word_masking does not work for RoBERTa
    mlm_prob = 0.15
    # Image-Text Matching
    draw_false_image = 1
    draw_false_text = 0 

    # Downstream Settings
    # Image-Text Recall
    get_recall_metric = False
    # Visual Question Answering
    vqav2_label_size = 3129

    # RefCOCO label size
    refcoco_label_size = 75

    #Bounding box weights
    giou_weight = 2.0
    cardinality_weight = 5.0
    entropy_weight = 1.0

    # Optimizer Setting
    optim_type = "adamw"
    learning_rate = 1e-5
    weight_decay = 0.01
    decay_power = 1
    max_epoch = 100
    max_steps = 100000
    warmup_steps = 10000
    end_lr = 0
    lr_mult_head = 5  # multiply lr for downstream heads
    lr_mult_cross_modal = 5  # multiply lr for the cross-modal module
    
    # PL Trainer Setting
    fast_dev_run = False
    val_check_interval = 1.0
    
    run_training = True
    run_test = True

    # below params varies with the environment
    data_root = 'data/arrow/' 
    log_dir = "result"
    csv_log_file = None
    num_gpus = 2
    num_nodes = 1
    num_workers = 12
    precision = 32

    # WAC distribution and embedding settings
    use_wac_embeddings = False
    wac_distribution_matrix = None
    wac_image_encoder = None
    use_wac_models_only = False
    pretrained_wac_embedding_file = None

    # WAC model settings
    position_size = 7
    neg_to_pos = None
    wac_kwargs = None
    num_cores = None
    wac_train_steps = 5
    wac_repo_id = None
    local_wac_directory = None
    save_wac_features = False

    # HuggingFace settings to save and upload model
    huggingface_save_directory = None
    huggingface_save_name = None
    push_to_hub = False
    subsection_to_save = None

# ===================== Experiment 2 Cofigs ===================== #

@ex.named_config
def pretrain_mlm_itm_twotower_exp2_deittiny_electrasmall():
    exp_name = "mlm_itm_twotower_exp2_deittiny_electrasmall"
    model_type = "two-tower"
    datasets = ["coco", "vg"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 512
    per_gpu_batchsize = 128
    max_epoch = None
    max_steps = 50000
    warmup_steps = 0.1
    whole_word_masking = True
    model_type = "two-tower"
    # DO NOT Freeze Encoders
    freeze_image_encoder = False
    freeze_text_encoder = False
    # Image settings
    image_encoder = "facebook/deit-tiny-distilled-patch16-224"
    image_size = 224
    patch_size = 16
    draw_false_image = 1
    image_only = False
    # Text Setting
    text_encoder = "google/electra-small-discriminator"
    max_text_len = 50
    whole_word_masking = True # note that whole_word_masking does not work for RoBERTa
    mlm_prob = 0.15
    draw_false_text = 0
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 6
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1
    # Optimizer Settings
    learning_rate = 1e-4
    val_check_interval = 1.0
    lr_mult_head = 5
    lr_mult_cross_modal = 5

@ex.named_config
def pretrain_mlm_onetower_electrasmall():
    exp_name = "mlm_onetower_electrasmall"
    model_type = "two-tower"
    datasets = ["babylm"]
    loss_names = _loss_names({"mlm": 1})
    batch_size = 128
    per_gpu_batchsize = 128
    max_epoch = 10
    max_steps = 1000000
    warmup_steps = 0.1
    whole_word_masking = False
    # DO NOT Freeze Encoders
    freeze_image_encoder = False
    freeze_text_encoder = False
    # Text Setting
    text_encoder = "bert_wac"
    random_init_text_encoder = True
    tokenizer = "google/electra-small-discriminator"
    text_encoder = "google/electra-small-discriminator"
    max_text_len = 50
    whole_word_masking = False # note that whole_word_masking does not work for RoBERTa
    mlm_prob = 0.15
    draw_false_text = 0
    draw_false_image = 0
    num_gpus = 1
    data_root = "data/arrow/babylm"

    # Image settings to not use image encoders
    use_image_encoder = False

    # WAC model settings
    # wac_embedding_size = 512
    # wac_distribution_matrix = "value"
    # wac_image_encoder = "openai/clip-vit-base-patch16"
    # save_directory = "wac_models"
    # wac_train_steps = 5
    # num_cores = 0

    # HuggingFace settings
    huggingface_save_directory = "results/huggingface_outputs"
    huggingface_save_name = "ajtorek/electra-renaissance-babylm"
    subsection_to_save = "text_transformer"
    push_to_hub = True

@ex.named_config
def pretrain_mlm_onetower_electrasmall_embedding520():
    exp_name = "mlm_onetower_electrasmall_embedding520"
    model_type = "two-tower"
    datasets = ["babylm"]
    loss_names = _loss_names({"mlm": 1})
    batch_size = 128
    per_gpu_batchsize = 128
    max_epoch = 10
    max_steps = 1000000
    warmup_steps = 0.1
    whole_word_masking = False
    # DO NOT Freeze Encoders
    freeze_image_encoder = False
    freeze_text_encoder = False
    # Text Setting
    text_encoder = "bert_wac"
    random_init_text_encoder = True
    text_encoder_manual_configuration = True
    tokenizer = "google/electra-small-discriminator"
    text_encoder = "google/electra-small-wac-discriminator"
    max_text_len = 50
    text_encoder_embedding_size = 520
    whole_word_masking = False # note that whole_word_masking does not work for RoBERTa
    mlm_prob = 0.15
    draw_false_text = 0
    draw_false_image = 0
    num_gpus = 1
    data_root = "data/arrow/babylm"

    # Image settings to not use image encoders
    use_image_encoder = False

    # HuggingFace settings
    huggingface_save_directory = "results/huggingface_outputs"
    huggingface_save_name = "ajtorek/electra-renaissance-babylm-embedding520"
    subsection_to_save = "text_transformer"
    push_to_hub = True

@ex.named_config
def pretrain_mim_onetower_deit_tiny():
    exp_name = "pretrain_mim_onetower_deit_tiny"
    model_type = "two-tower"
    datasets = ["sbu"]
    loss_names = _loss_names({"mim": 1})
    batch_size = 256
    per_gpu_batchsize = 256
    max_epoch = 20
    max_steps = 1000000
    warmup_steps = 0.1
    whole_word_masking = False
    num_gpus = 1
    data_root = "data/arrow/sbucaptions"

    use_text_encoder = False

    use_image_encoder = True
    random_init_vision_encoder = True
    image_encoder_manual_configuration = True
    image_encoder = "facebook/deit-tiny-patch16-224"
    image_encoder_mlp_ratio = 1

    huggingface_save_directory = "results/huggingface_outputs"
    huggingface_save_name = "ajtorek/deit-tiny-renaissance"
    subsection_to_save = "image_encoder"
    push_to_hub = True

@ex.named_config
def pretrain_wac_itm_twotower_electrasmall_deit_small():
    exp_name = "wac_itm_twotower_electrasmall_deit_small"
    model_type = "two-tower"
    datasets = ["coco"]
    loss_names = _loss_names({"itm": 1})
    batch_size = 256
    per_gpu_batchsize = 256
    max_epoch = 20
    warmup_steps = 0.1
    whole_word_masking = False
    # DO NOT Freeze Encoders
    freeze_image_encoder = False
    freeze_text_encoder = False
    # Text Setting
    text_encoder = "ajtorek/electra-wac-renaissance-babylm"
    random_init_text_encoder = False
    tokenizer = "google/electra-small-discriminator"
    max_text_len = 50
    whole_word_masking = False # note that whole_word_masking does not work for RoBERTa
    mlm_prob = 0.15
    draw_false_text = 0
    draw_false_image = 1
    num_gpus = 1
    data_root = "data/arrow/coco"

    # Image settings to not use image encoders
    use_image_encoder = True
    image_encoder = "facebook/deit-small-patch16-224"

    # Cross Layer Settings
    cross_layer_hidden_size = 320
    num_cross_layers = 6
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1

    # WAC model settings
    use_wac_embeddings = False
    wac_distribution_matrix = None
    wac_image_encoder = None
    #wac_train_steps = 5
    num_cores = 0

    # Training settings
    learning_rate = 1e-4

    # HuggingFace settings
    huggingface_save_directory = "results/huggingface_outputs"
    huggingface_save_name = "ajtorek/electra-deit-itm-renaissance-wac"
    push_to_hub = True

    # Two-tower model settings
    learning_rate = 1e-4

@ex.named_config
def pretrain_wac_ref_twotower_electrasmall_deit_small():
    exp_name = "wac_ref_twotower_electrasmall_deit_small"
    model_type = "two-tower"
    datasets = ["coco"]
    loss_names = _loss_names({"ref": 1})
    batch_size = 256
    per_gpu_batchsize = 256
    max_epoch = 20
    warmup_steps = 0.1
    whole_word_masking = False
    # DO NOT Freeze Encoders
    freeze_image_encoder = False
    freeze_text_encoder = False
    # Text Setting
    text_encoder = "ajtorek/electra-wac-renaissance-babylm"
    random_init_text_encoder = False
    tokenizer = "google/electra-small-discriminator"
    max_text_len = 50
    whole_word_masking = False # note that whole_word_masking does not work for RoBERTa
    mlm_prob = 0.15
    draw_false_text = 0
    draw_false_image = 1
    num_gpus = 1
    data_root = "data/arrow/coco"

    # Image settings to not use image encoders
    use_image_encoder = True
    image_encoder = "facebook/deit-small-patch16-224"

    # Cross Layer Settings
    cross_layer_hidden_size = 320
    num_cross_layers = 6
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1

    # WAC model settings
    use_wac_embeddings = False
    wac_distribution_matrix = None
    wac_image_encoder = None
    #wac_train_steps = 5
    num_cores = 0

    # Training settings
    learning_rate = 1e-4

    # HuggingFace settings
    huggingface_save_directory = "results/huggingface_outputs"
    huggingface_save_name = "ajtorek/electra-deit-ref-renaissance-wac"
    push_to_hub = True

    # Two-tower model settings
    learning_rate = 1e-4

@ex.named_config
def pretrain_wac_ref_bbox_twotower_electrasmall_deit_small():
    exp_name = "wac_ref_bbox_twotower_electrasmall_deit_small"
    model_type = "two-tower"
    datasets = ["coco"]
    loss_names = _loss_names({"ref_bbox": 1})
    batch_size = 256
    per_gpu_batchsize = 256
    max_epoch = 20
    warmup_steps = 0.1
    whole_word_masking = False
    # DO NOT Freeze Encoders
    freeze_image_encoder = False
    freeze_text_encoder = False
    # Text Setting
    text_encoder = "ajtorek/electra-wac-renaissance-babylm"
    random_init_text_encoder = False
    tokenizer = "google/electra-small-discriminator"
    max_text_len = 50
    whole_word_masking = False # note that whole_word_masking does not work for RoBERTa
    mlm_prob = 0.15
    draw_false_text = 0
    draw_false_image = 1
    num_gpus = 1
    data_root = "data/arrow/coco"
    

    # Image settings to not use image encoders
    use_image_encoder = True
    image_encoder = "facebook/deit-small-patch16-224"

    # Cross Layer Settings
    cross_layer_hidden_size = 320
    num_cross_layers = 6
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1

    # WAC model settings
    use_wac_embeddings = False
    wac_distribution_matrix = None
    wac_image_encoder = None
    #wac_train_steps = 5
    num_cores = 0

    # Training settings
    learning_rate = 1e-4

    # HuggingFace settings
    huggingface_save_directory = "results/huggingface_outputs"
    huggingface_save_name = "ajtorek/electra-deit-ref-bbox-renaissance-wac"
    push_to_hub = True
    
@ex.named_config
def eval_ref_twotower_electrasmall_deit_small():
    exp_name = "eval_ref_twotower_electrasmall_deit_small"
    model_type = "two-tower"
    datasets = ["coco"]
    loss_names = _loss_names({"ref": 1})
    num_gpus = 1
    precision = 'bf16-mixed'
    batch_size = 8
    per_gpu_batchsize = 8
    
    max_epoch = 5
    warmup_steps = 0.1
    whole_word_masking = False
    # DO NOT Freeze Encoders
    freeze_image_encoder = False
    freeze_text_encoder = False
    # Text Setting
    text_encoder = "ajtorek/electra-wac-renaissance-babylm"
    random_init_text_encoder = False
    tokenizer = "google/electra-small-discriminator"
    max_text_len = 50
    whole_word_masking = False # note that whole_word_masking does not work for RoBERTa
    mlm_prob = 0.15
    draw_false_text = 0
    draw_false_image = 0
    num_gpus = 1
    data_root = "data/arrow/coco"
    
    # Image settings to not use image encoders
    use_image_encoder = True
    image_encoder = "facebook/deit-small-patch16-224"

    # Cross Layer Settings
    cross_layer_hidden_size = 320
    num_cross_layers = 6
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1

    # WAC model settings
    use_wac_embeddings = False
    wac_distribution_matrix = None
    wac_image_encoder = None
    #wac_train_steps = 5
    num_cores = 0

    # HuggingFace settings
    huggingface_save_directory = None
    huggingface_save_name = None
    push_to_hub = False

@ex.named_config
def eval_ref_twotower_electrasmall_deit_wac_embeddings():
    exp_name = "eval_ref_twotower_electrasmall_wac_embeddings"
    model_type = "two-tower"
    datasets = ["coco"]
    loss_names = _loss_names({"ref": 1})
    num_gpus = 1
    precision = 32
    batch_size = 8
    per_gpu_batchsize = 8
    #complete_encoder_path = "ajtorek/electra-deit-itm-renaissance-wac"
    
    max_epoch = 5
    warmup_steps = 0.1
    whole_word_masking = False
    learning_rate = 1e-4
    # DO NOT Freeze Encoders
    freeze_image_encoder = False
    freeze_text_encoder = False
    # Text Setting
    text_encoder = "ajtorek/electra-wac-renaissance-babylm"
    random_init_text_encoder = True
    text_encoder_manual_configuration = True
    tokenizer = "google/electra-small-discriminator"
    max_text_len = 50
    text_encoder_embedding_size = 520
    whole_word_masking = False # note that whole_word_masking does not work for RoBERTa
    mlm_prob = 0.15
    draw_false_text = 0
    draw_false_image = 0
    num_gpus = 1
    data_root = "data/arrow/coco"
    
    # Image settings to not use image encoders
    use_image_encoder = True
    image_encoder = "facebook/deit-small-patch16-224"

    # Cross Layer Settings
    cross_layer_hidden_size = 320
    num_cross_layers = 6
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1

    # WAC model settings
    use_wac_embeddings = True
    wac_distribution_matrix = None
    wac_image_encoder = "openai/clip-vit-base-patch16"
    wac_repo_id = "ajtorek/wac_weights"
    local_wac_directory = "wac_models"
    #wac_train_steps = 5
    num_cores = 0
    save_wac_features = True

    # HuggingFace settings
    huggingface_save_directory = "models"
    huggingface_save_name = "ajtorek/electra_deit_small_wac_embeddings"
    push_to_hub = True

@ex.named_config
def eval_ref_twotower_electrasmall_deit_text_pretrained_wac_embeddings():
    exp_name = "eval_ref_twotower_electrasmall_wac_text_pretrained_embeddings"
    model_type = "two-tower"
    datasets = ["coco"]
    loss_names = _loss_names({"ref": 1})
    num_gpus = 1
    precision = 32
    batch_size = 8
    per_gpu_batchsize = 8
    #complete_encoder_path = "ajtorek/electra-deit-itm-renaissance-wac"
    
    max_epoch = 5
    warmup_steps = 0.1
    whole_word_masking = False
    learning_rate = 1e-4
    # DO NOT Freeze Encoders
    freeze_image_encoder = True
    freeze_text_encoder = False
    # Text Setting
    text_encoder = "ajtorek/electra-wac-renaissance-babylm"
    random_init_text_encoder = False
    text_encoder_manual_configuration = False
    text_encoder_hidden_size = 256
    tokenizer = "google/electra-small-discriminator"
    max_text_len = 50
    whole_word_masking = False # note that whole_word_masking does not work for RoBERTa
    mlm_prob = 0.15
    draw_false_text = 0
    draw_false_image = 0
    num_gpus = 1
    data_root = "data/arrow/coco"
    
    # Image settings to not use image encoders
    use_image_encoder = False
    image_encoder = "facebook/deit-small-patch16-224"
    use_wac_models_only = False
   
    # Cross Layer Settings
    cross_layer_hidden_size = 320
    num_cross_layers = 6
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1

    # WAC model settings
    use_wac_embeddings = True
    wac_distribution_matrix = None
    wac_image_encoder = "openai/clip-vit-base-patch16"
    wac_repo_id = "ajtorek/wac_weights"
    local_wac_directory = "wac_models"
    pretrained_wac_embedding_file = "pretrained_embeddings/clip.bertvocab.embeddings.128.txt"

    #wac_train_steps = 5
    num_cores = 0
    save_wac_features = True

    # HuggingFace settings
    huggingface_save_directory = "models"
    huggingface_save_name = "ajtorek/electra-deit-small-text-pretrained-wac-embeddings"
    push_to_hub = True

@ex.named_config
def eval_ref_twotower_electrasmall_deit_wac_distributions():
    exp_name = "eval_ref_twotower_electrasmall_wac_distributions"
    model_type = "two-tower"
    datasets = ["coco"]
    loss_names = _loss_names({"ref": 1})
    num_gpus = 1
    precision = 32
    batch_size = 8
    per_gpu_batchsize = 8
    complete_encoder_path = "ajtorek/electra-deit-itm-renaissance-wac"
    
    max_epoch = 5
    warmup_steps = 0.1
    whole_word_masking = False
    learning_rate = 1e-4
    # DO NOT Freeze Encoders
    freeze_image_encoder = False
    freeze_text_encoder = False
    # Text Setting
    text_encoder = "ajtorek/electra-wac-renaissance-babylm"
    random_init_text_encoder = False
    tokenizer = "google/electra-small-discriminator"
    max_text_len = 50
    whole_word_masking = False # note that whole_word_masking does not work for RoBERTa
    mlm_prob = 0.15
    draw_false_text = 0
    draw_false_image = 0
    num_gpus = 1
    data_root = "data/arrow/coco"
    
    # Image settings to not use image encoders
    use_image_encoder = True
    image_encoder = "facebook/deit-small-patch16-224"

    # Cross Layer Settings
    cross_layer_hidden_size = 320
    num_cross_layers = 6
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 7
    cross_layer_drop_rate = 0.1

    # WAC model settings
    use_wac_embeddings = False
    wac_distribution_matrix = "value"
    wac_image_encoder = "openai/clip-vit-base-patch16"
    #wac_train_steps = 5
    wac_repo_id = "ajtorek/wac_weights"
    local_wac_directory = "wac_models"
    num_cores = 0
    save_wac_features = True

    # HuggingFace settings
    huggingface_save_directory = "models"
    huggingface_save_name = "ajtorek/electra_deit_small_wac_distributions"
    push_to_hub = True

@ex.named_config
def finetune_cola_onetower_electrasmall():
    exp_name = "finetune_cola_onetower_electrasmall"
    model_type = "two-tower"
    datasets = ["glue"]
    loss_names = _loss_names({"cola":1})
    csv_log_file = "glue_results/glue.csv"
    complete_encoder_path = "ajtorek/electra-deit-small-text-pretrained-wac-embeddings"
    
    batch_size = 128
    per_gpu_batchsize = 128
    max_epoch = 10
    warmup_steps = 0.1
    learning_rate = 5e-5
    whole_word_masking = False
    # DO NOT Freeze Encoders
    freeze_image_encoder = True
    freeze_text_encoder = False
    random_init_text_encoder = False
    # Text Setting
    text_encoder = "ajtorek/electra-wac-renaissance-babylm"
    tokenizer = "google/electra-small-discriminator"
    max_text_len = 50
    whole_word_masking = False # note that whole_word_masking does not work for RoBERTa
    mlm_prob = 0.15
    draw_false_text = 0
    draw_false_image = 0
    num_gpus = 1
    data_root = "data/arrow/glue/mnli"

    # Image settings to not use image encoders
    use_image_encoder = True
    image_encoder = "facebook/deit-small-patch16-224"

    # Cross Layer Settings
    cross_layer_hidden_size = 320
    num_cross_layers = 6
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1

    # WAC model settings
    use_wac_embeddings = True
    wac_distribution_matrix = None
    wac_image_encoder = "openai/clip-vit-base-patch16"
    #wac_train_steps = 5
    wac_repo_id = "ajtorek/wac_weights"
    local_wac_directory = "wac_models"
    num_cores = 0
    save_wac_features = False
    pretrained_wac_embedding_file = "pretrained_embeddings/clip.bertvocab.embeddings.128.txt"


@ex.named_config
def finetune_mnli_onetower_electrasmall():
    exp_name = "finetune_mnli_onetower_electrasmall"
    model_type = "two-tower"
    datasets = ["glue"]
    loss_names = _loss_names({"mnli": 1})
    csv_log_file = "glue_results/glue.csv"
    complete_encoder_path = "ajtorek/electra-deit-small-text-pretrained-wac-embeddings"

    batch_size = 128
    per_gpu_batchsize = 128
    max_epoch = 10
    warmup_steps = 0.1
    learning_rate = 5e-5
    whole_word_masking = False
    # DO NOT Freeze Encoders
    freeze_image_encoder = True
    freeze_text_encoder = False
    random_init_text_encoder = False
    # Text Setting
    text_encoder = "ajtorek/electra-wac-renaissance-babylm"
    tokenizer = "google/electra-small-discriminator"
    max_text_len = 50
    whole_word_masking = False # note that whole_word_masking does not work for RoBERTa
    mlm_prob = 0.15
    draw_false_text = 0
    draw_false_image = 0
    num_gpus = 1
    data_root = "data/arrow/glue/mnli"

    # Image settings to not use image encoders
    use_image_encoder = True
    image_encoder = "facebook/deit-small-patch16-224"

    # Cross Layer Settings
    cross_layer_hidden_size = 320
    num_cross_layers = 6
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1

    # WAC model settings
    use_wac_embeddings = True
    wac_distribution_matrix = None
    wac_image_encoder = "openai/clip-vit-base-patch16"
    #wac_train_steps = 5
    wac_repo_id = "ajtorek/wac_weights"
    local_wac_directory = "wac_models"
    num_cores = 0
    save_wac_features = False
    pretrained_wac_embedding_file = "pretrained_embeddings/clip.bertvocab.embeddings.128.txt"
    

@ex.named_config
def finetune_mrpc_onetower_electrasmall():
    exp_name = "finetune_mnli_onetower_electrasmall"
    model_type = "two-tower"
    datasets = ["glue"]
    loss_names = _loss_names({"mrpc": 1})
    csv_log_file = "glue_results/glue.csv"
    complete_encoder_path = "ajtorek/electra-deit-small-text-pretrained-wac-embeddings"

    batch_size = 128
    per_gpu_batchsize = 128
    max_epoch = 10
    warmup_steps = 0.1
    learning_rate = 5e-5
    whole_word_masking = False
   
    # DO NOT Freeze Encoders
    freeze_image_encoder = True
    freeze_text_encoder = False
    random_init_text_encoder = False
    # Text Setting
    text_encoder = "ajtorek/electra-wac-renaissance-babylm"
    tokenizer = "google/electra-small-discriminator"
    max_text_len = 50
    whole_word_masking = False # note that whole_word_masking does not work for RoBERTa
    mlm_prob = 0.15
    draw_false_text = 0
    draw_false_image = 0
    num_gpus = 1
    data_root = "data/arrow/glue/mnli"

    # Image settings to not use image encoders
    use_image_encoder = True
    image_encoder = "facebook/deit-small-patch16-224"

    # Cross Layer Settings
    cross_layer_hidden_size = 320
    num_cross_layers = 6
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1

    # WAC model settings
    use_wac_embeddings = True
    wac_distribution_matrix = None
    wac_image_encoder = "openai/clip-vit-base-patch16"
    #wac_train_steps = 5
    wac_repo_id = "ajtorek/wac_weights"
    local_wac_directory = "wac_models"
    num_cores = 0
    save_wac_features = False
    pretrained_wac_embedding_file = "pretrained_embeddings/clip.bertvocab.embeddings.128.txt"

@ex.named_config
def finetune_qqp_onetower_electrasmall():
    exp_name = "finetune_qqp_onetower_electrasmall"
    model_type = "two-tower"
    datasets = ["glue"]
    loss_names = _loss_names({"qqp": 1})
    csv_log_file = "glue_results/glue.csv"
    complete_encoder_path = "ajtorek/electra-deit-small-text-pretrained-wac-embeddings"

    batch_size = 128
    per_gpu_batchsize = 128
    max_epoch = 10
    warmup_steps = 0.1
    learning_rate = 5e-5
    whole_word_masking = False
    # DO NOT Freeze Encoders
    freeze_image_encoder = True
    freeze_text_encoder = False
    random_init_text_encoder = False
    # Text Setting
    text_encoder = "ajtorek/electra-wac-renaissance-babylm"
    tokenizer = "google/electra-small-discriminator"
    max_text_len = 50
    whole_word_masking = False # note that whole_word_masking does not work for RoBERTa
    mlm_prob = 0.15
    draw_false_text = 0
    draw_false_image = 0
    num_gpus = 1
    data_root = "data/arrow/glue/mnli"

    # Image settings to not use image encoders
    use_image_encoder = True
    image_encoder = "facebook/deit-small-patch16-224"

    # Cross Layer Settings
    cross_layer_hidden_size = 320
    num_cross_layers = 6
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1

    # WAC model settings
    use_wac_embeddings = True
    wac_distribution_matrix = None
    wac_image_encoder = "openai/clip-vit-base-patch16"
    #wac_train_steps = 5
    wac_repo_id = "ajtorek/wac_weights"
    local_wac_directory = "wac_models"
    num_cores = 0
    save_wac_features = False
    pretrained_wac_embedding_file = "pretrained_embeddings/clip.bertvocab.embeddings.128.txt"

@ex.named_config
def finetune_qnli_onetower_electrasmall():
    exp_name = "finetune_qnli_onetower_electrasmall"
    model_type = "two-tower"
    datasets = ["glue"]
    loss_names = _loss_names({"qnli": 1})
    csv_log_file = "glue_results/glue.csv"
    complete_encoder_path = "ajtorek/electra-deit-small-text-pretrained-wac-embeddings"

    batch_size = 128
    per_gpu_batchsize = 128
    max_epoch = 10
    warmup_steps = 0.1
    learning_rate = 5e-5
    whole_word_masking = False

    # DO NOT Freeze Encoders
    freeze_image_encoder = True
    freeze_text_encoder = False
    random_init_text_encoder = False
    # Text Setting
    text_encoder = "ajtorek/electra-wac-renaissance-babylm"
    tokenizer = "google/electra-small-discriminator"
    max_text_len = 50
    whole_word_masking = False # note that whole_word_masking does not work for RoBERTa
    mlm_prob = 0.15
    draw_false_text = 0
    draw_false_image = 0
    num_gpus = 1
    data_root = "data/arrow/glue/mnli"

    # Image settings to not use image encoders
    use_image_encoder = True
    image_encoder = "facebook/deit-small-patch16-224"

    # Cross Layer Settings
    cross_layer_hidden_size = 320
    num_cross_layers = 6
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1

    # WAC model settings
    use_wac_embeddings = True
    wac_distribution_matrix = None
    wac_image_encoder = "openai/clip-vit-base-patch16"
    #wac_train_steps = 5
    wac_repo_id = "ajtorek/wac_weights"
    local_wac_directory = "wac_models"
    num_cores = 0
    save_wac_features = False
    pretrained_wac_embedding_file = "pretrained_embeddings/clip.bertvocab.embeddings.128.txt"

@ex.named_config
def finetune_rte_onetower_electrasmall():
    exp_name = "finetune_rte_onetower_electrasmall"
    model_type = "two-tower"
    datasets = ["glue"]
    loss_names = _loss_names({"rte": 1})
    csv_log_file = "glue_results/glue.csv"
    complete_encoder_path = "ajtorek/electra-deit-small-text-pretrained-wac-embeddings"

    batch_size = 128
    per_gpu_batchsize = 128
    max_epoch = 10
    warmup_steps = 0.1
    learning_rate = 5e-5
    whole_word_masking = False

    # DO NOT Freeze Encoders
    freeze_image_encoder = True
    freeze_text_encoder = False
    random_init_text_encoder = False
    # Text Setting
    text_encoder = "ajtorek/electra-wac-renaissance-babylm"
    tokenizer = "google/electra-small-discriminator"
    max_text_len = 50
    whole_word_masking = False # note that whole_word_masking does not work for RoBERTa
    mlm_prob = 0.15
    draw_false_text = 0
    draw_false_image = 0
    num_gpus = 1
    data_root = "data/arrow/glue/mnli"

    # Image settings to not use image encoders
    use_image_encoder = True
    image_encoder = "facebook/deit-small-patch16-224"

    # Cross Layer Settings
    cross_layer_hidden_size = 320
    num_cross_layers = 6
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1

    # WAC model settings
    use_wac_embeddings = True
    wac_distribution_matrix = None
    wac_image_encoder = "openai/clip-vit-base-patch16"
    #wac_train_steps = 5
    wac_repo_id = "ajtorek/wac_weights"
    local_wac_directory = "wac_models"
    num_cores = 0
    save_wac_features = False
    pretrained_wac_embedding_file = "pretrained_embeddings/clip.bertvocab.embeddings.128.txt"

@ex.named_config
def finetune_sst2_onetower_electrasmall():
    exp_name = "finetune_sst2_onetower_electrasmall"
    model_type = "two-tower"
    datasets = ["glue"]
    loss_names = _loss_names({"sst2": 1})
    csv_log_file = "glue_results/glue.csv"
    complete_encoder_path = "ajtorek/electra-deit-small-text-pretrained-wac-embeddings"

    batch_size = 128
    per_gpu_batchsize = 128
    max_epoch = 10
    warmup_steps = 0.1
    learning_rate = 5e-5
    whole_word_masking = False

    # DO NOT Freeze Encoders
    freeze_image_encoder = True
    freeze_text_encoder = False
    random_init_text_encoder = False
    # Text Setting
    text_encoder = "ajtorek/electra-wac-renaissance-babylm"
    tokenizer = "google/electra-small-discriminator"
    max_text_len = 50
    whole_word_masking = False # note that whole_word_masking does not work for RoBERTa
    mlm_prob = 0.15
    draw_false_text = 0
    draw_false_image = 0
    num_gpus = 1
    data_root = "data/arrow/glue/mnli"

    # Image settings to not use image encoders
    use_image_encoder = True
    image_encoder = "facebook/deit-small-patch16-224"

    # Cross Layer Settings
    cross_layer_hidden_size = 320
    num_cross_layers = 6
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1

    # WAC model settings
    use_wac_embeddings = True
    wac_distribution_matrix = None
    wac_image_encoder = "openai/clip-vit-base-patch16"
    #wac_train_steps = 5
    wac_repo_id = "ajtorek/wac_weights"
    local_wac_directory = "wac_models"
    num_cores = 0
    save_wac_features = False
    pretrained_wac_embedding_file = "pretrained_embeddings/clip.bertvocab.embeddings.128.txt"

@ex.named_config
def finetune_stsb_onetower_electrasmall():
    exp_name = "finetune_stsb_onetower_electrasmall"
    model_type = "two-tower"
    datasets = ["glue"]
    loss_names = _loss_names({"stsb": 1})
    csv_log_file = "glue_results/glue.csv"
    complete_encoder_path = "ajtorek/electra-deit-small-text-pretrained-wac-embeddings"

    batch_size = 128
    per_gpu_batchsize = 128
    max_epoch = 10
    warmup_steps = 0.1
    learning_rate = 5e-5
    whole_word_masking = False
    # DO NOT Freeze Encoders
    freeze_image_encoder = True
    freeze_text_encoder = False
    random_init_text_encoder = False
    # Text Setting
    text_encoder = "ajtorek/electra-wac-renaissance-babylm"
    tokenizer = "google/electra-small-discriminator"
    max_text_len = 50
    whole_word_masking = False # note that whole_word_masking does not work for RoBERTa
    mlm_prob = 0.15
    draw_false_text = 0
    draw_false_image = 0
    num_gpus = 1
    data_root = "data/arrow/glue/mnli"

    # Image settings to not use image encoders
    use_image_encoder = True
    image_encoder = "facebook/deit-small-patch16-224"

    # Cross Layer Settings
    cross_layer_hidden_size = 320
    num_cross_layers = 6
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1

    # WAC model settings
    use_wac_embeddings = True
    wac_distribution_matrix = None
    wac_image_encoder = "openai/clip-vit-base-patch16"
    #wac_train_steps = 5
    wac_repo_id = "ajtorek/wac_weights"
    local_wac_directory = "wac_models"
    num_cores = 0
    save_wac_features = False
    pretrained_wac_embedding_file = "pretrained_embeddings/clip.bertvocab.embeddings.128.txt"

@ex.named_config
def finetune_wnli_onetower_electrasmall():
    exp_name = "finetune_wnli_onetower_electrasmall"
    model_type = "two-tower"
    datasets = ["glue"]
    loss_names = _loss_names({"wnli": 1})
    csv_log_file = "glue_results/glue.csv"
    complete_encoder_path = "ajtorek/electra-deit-small-text-pretrained-wac-embeddings"

    batch_size = 128
    per_gpu_batchsize = 128
    max_epoch = 10
    warmup_steps = 0.1
    learning_rate = 5e-5
    whole_word_masking = False

    # DO NOT Freeze Encoders
    freeze_image_encoder = False
    freeze_text_encoder = False
    random_init_text_encoder = False
    # Text Setting
    text_encoder = "ajtorek/electra-wac-renaissance-babylm"
    tokenizer = "google/electra-small-discriminator"
    max_text_len = 50
    whole_word_masking = False # note that whole_word_masking does not work for RoBERTa
    mlm_prob = 0.15
    draw_false_text = 0
    draw_false_image = 0
    num_gpus = 1
    data_root = "data/arrow/glue/mnli"

    # Image settings to not use image encoders
    use_image_encoder = True
    image_encoder = "facebook/deit-small-patch16-224"

    # Cross Layer Settings
    cross_layer_hidden_size = 320
    num_cross_layers = 6
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1

    # WAC model settings
    use_wac_embeddings = True
    wac_distribution_matrix = None
    wac_image_encoder = "openai/clip-vit-base-patch16"
    #wac_train_steps = 5
    wac_repo_id = "ajtorek/wac_weights"
    local_wac_directory = "wac_models"
    num_cores = 0
    save_wac_features = False
    pretrained_wac_embedding_file = "pretrained_embeddings/clip.bertvocab.embeddings.128.txt"