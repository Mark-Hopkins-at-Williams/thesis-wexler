from configure import USE_CUDA
import gc
import sys
import torch
from transformers import AutoModelForSeq2SeqLM, AutoConfig


def logger(s, to_stderr=False):
    if to_stderr:
        sys.stderr.write(s + "\n")
        sys.stderr.flush()
    else:
        sys.stdout.write(s + "\n")
        sys.stdout.flush()


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


## PREPARE MODEL FOR TRAINING ACCORDING TO EXPERIMENT SPECIFICATIONS
def prepare_model_for_finetuning(ft_params):
    if ft_params.should_finetune:
        model = AutoModelForSeq2SeqLM.from_pretrained(ft_params.base_model)
        print("loaded pretrained model")
    else:
        model_config = AutoConfig.from_pretrained(ft_params.base_model)

        model_config.d_model = ft_params.d_model
        model_config.encoder_ffn_dim = ft_params.d_ff
        model_config.decoder_ffn_dim = ft_params.d_ff
        model_config.encoder_layers = ft_params.encoder_layers
        model_config.decoder_layers = ft_params.decoder_layers

        model = AutoModelForSeq2SeqLM.from_config(model_config)
        print("loaded architecture only")
    if hasattr(model.config, "max_length"):  # this should be in a GenerationConfig
        delattr(model.config, "max_length")
    if ft_params.freeze_decoder:
        print("--> DECODER FROZEN <--")
        for param in model.get_decoder().parameters():
            param.requires_grad = False
    else:
        print("--> decoder NOT frozen <--")
    if ft_params.freeze_encoder:
        print("--> ENCODER FROZEN <--")
        for param in model.get_encoder().parameters():
            param.requires_grad = False
    else:
        print("--> encoder NOT frozen <--")
    if USE_CUDA:
        torch.cuda.set_device(0)
        model.cuda()
    return model
