#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: dev_nb/00_utils_nb.ipynb
#import pandas as pd
#import numpy as np
from typing import List
#from fastai.text.all import *
from fastai.callback import *
from fastinference.inference import *
from .generative_basics import *
from fcd_torch import *
from random import choices
from guacamol.distribution_learning_benchmark import ValidityBenchmark, UniquenessBenchmark, NoveltyBenchmark
from guacamol.frechet_benchmark import FrechetBenchmark
from guacamol.distribution_matching_generator import DistributionMatchingGenerator

# if torch.cuda.is_available():
    
#     fcd = FCD(device='cuda:0', n_jobs=8)

# else:
#     fcd = FCD(device='cpu',n_jobs=8)
    
class MockGenerator(DistributionMatchingGenerator):
    """
    Mock generator that returns pre-defined molecules,
    possibly split in several calls
    """

    def __init__(self, molecules) -> None:
        self.molecules = molecules
        self.cursor = 0

    def generate(self, number_samples: int):
        end = self.cursor + number_samples

        sampled_molecules = self.molecules[self.cursor:end]
        self.cursor = end
        return sampled_molecules

class GenerativeCallback(Callback):

    def __init__(self, reference_mols:collections=[], text:str='', max_size:int=100, temperature:float=0.7, max_mols:int=100):
        super().__init__()
        self.reference_mols = reference_mols
        self.text = text
        self.max_size = max_size
        self.temperature = temperature
        self.max_mols = max_mols
        self.smiles = []
        self.valid_mols = []
        
        # Define the benchmark before training because it needs to calculate the mean and covariance for ref mols
        self.fcd_benchmark = FrechetBenchmark(training_set=reference_mols, sample_size=len(reference_mols))
 
    def sampling(self):

        self.model.reset()    # Reset the model

        nums = self.dls.numericalize
        stop_index = self.dls.train.vocab.index(BOS)

        idxs = idxs_all = self.dls.test_dl([self.text]).items[0].to(self.dls.device)
        for _ in range(self.max_size):
            preds = self.get_preds(dl=[(idxs[None],)], decoded_loss=False)

            res = tensor(preds[0][0][-1])
            #print(res.shape)
            if self.temperature != 1.: res.pow_(1 / self.temperature)
            idx = torch.multinomial(res, 1).item()
            if idx != stop_index:

                idxs = idxs_all = torch.cat([idxs_all, idxs.new([idx])])
            else:
                break
        decoded = ''.join([nums.vocab[o] for o in idxs_all if nums.vocab[o] not in [BOS, PAD]])  # Decode predicted tokens
        return decoded
    

    def _validity_score(self):
        gen = MockGenerator(self.smiles)
        val = ValidityBenchmark(number_samples=len(gen.molecules)).assess_model(gen).score        
        return val
    
    def _fcd_score(self):
        gen = MockGenerator(self.smiles)
        fcd_score = self.fcd_benchmark.assess_model(gen).score
        return fcd_score

    def _uniqueness_score(self): 
        gen = MockGenerator(self.smiles)
        unq = UniquenessBenchmark(number_samples=len(gen.molecules)).assess_model(gen).score
        return unq

    def _novelty_score(self):
        gen = MockGenerator(self.smiles)
        nov = NoveltyBenchmark(number_samples=len(gen.molecules),training_set=self.reference_mols).assess_model(gen).score
        return nov       

    def before_epoch(self):
        self.val, self.unq, self.nov = 0, 0, 0
        self.smiles = []
        self.valid_mols = []

    def before_validate(self, **kwargs):

        self.smiles += [self.sampling() for _ in range(self.max_mols)]
        print(self.smiles[:5])

#         self.valid_mols += [MolToSmiles(MolFromSmiles(x)) for x in self.smiles if MolFromSmiles(x)]
#         print(self.valid_mols[:5])
#         print(len(self.valid_mols))