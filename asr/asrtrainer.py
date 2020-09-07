#!/usr/bin/env python

import os

import copy
from omegaconf import DictConfig
from ruamel.yaml import YAML


import pytorch_lightning as pl
from nemo.collections.asr.models.ctc_models import EncDecCTCModel
import nemo.collections.asr as nemo_asr
asrmtype = EncDecCTCModel

if os.environ['KALDI_ROOT'].startswith("/mnt/matylda5"):
    os.environ['KALDI_ROOT'] = '/home/jaganadhg/AI_RND/nvidianemo/kaldi/src/'
""" 
Edit this to match with you Kaladi configuration.
Even if this configuration is not there systems will work
"""


def train_model(model_config: dict, train_manifest: str, val_manifest: str,
                max_epochs: int = 5, optimize: bool = False) -> None:
    """ Train an ASR model with Newmo
        Parameters
        -----------
        model_config : Configurations of an ASR model 
        train_manifest : path to the training manifest in NEMO manifest format 
        val_manifest : Path to validation manifest file in NEMO format 
        max_epochs : max epochs for training
        optimize : IF set True the model is optmized after traiing
    """
    
    asr_trainer = pl.Trainer(gpus=model_config['trainer']['gpus'],
                             max_epochs=max_epochs)
    
    model_config['model']['train_ds']['manifest_filepath'] = train_manifest
    model_config['model']['validation_ds']['manifest_filepath'] = val_manifest
    
    asr_model = nemo_asr.models.EncDecCTCModel(
        cfg=DictConfig(model_config['model']),
        trainer=asr_trainer
        )
    
    asr_trainer.fit(asr_model)
    
    if optimize is True:
        optimize_params = copy.deepcopy(model_config['model']['optim'])
        optimize_params['lr'] = 0.001
        asr_model.setup_optimization(optim_config=DictConfig(optimize_params))
        
        asr_trainer.fit(asr_model)
        
    return asr_model


def computer_wer(model_config: dict, asr_model: asrmtype) -> float:
    """ Computer WER from ASR Model
        Parameters
        ----------
        model_config : Model configuration
        asr_model : Asr Model
        
        Returns
        ---------
        wer_rate : Word Error Rate
        
    """
    
    model_config['model']['validation_ds']['batch_size'] = 2
    # Using small batch size to avoid cuds memory error local test 
    
    asr_model.setup_test_data(
        test_data_config=model_config['model']['validation_ds']
        )
    asr_model.cuda()
    
    wer_numerators = list()
    wer_denominators = list()
    
    for test_btch in asr_model.test_dataloader():
        test_batch = [wav.cuda() for wav in test_btch]
        targets = test_batch[2]
        
        targets_lengths = test_batch[3]        
        log_probs, encoded_len, greedy_predictions = asr_model(
                input_signal=test_batch[0],
                input_signal_length=test_batch[1]
            )
        
        wer_numerator, wer_denominator = asr_model._wer(greedy_predictions, 
                                                        targets, 
                                                        targets_lengths)
        wer_numerators.append(wer_numerator.detach().cpu().numpy())
        wer_denominators.append(wer_denominator.detach().cpu().numpy())
        
    wer_rate = sum(wer_numerators)/sum(wer_denominators)
    
    print(f"WER is {wer_rate} ")
    
    return wer_rate
    

if __name__ == "__main__":
    config_path = 'quartznet_15x5.yaml'
    train_manfest = "metadata.json"
    val_manifest = "metadata_validation.json"
    yaml = YAML(typ='safe')
    with open(config_path) as f:
        model_params = yaml.load(f)
        
    my_asr_model = train_model(model_params,
                               train_manfest,
                               val_manifest,
                               max_epochs=500,
                               optimize=False)
    
    wer = computer_wer(model_params,
                       my_asr_model)




