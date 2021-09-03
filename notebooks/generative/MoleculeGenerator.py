from generative.generative_basics import *
from rdkit.Chem import MolFromSmiles, MolToSmiles
from tqdm.notebook import tqdm

def is_valid(smiles):
    if isinstance(smiles, str): 
        mol = MolFromSmiles(smiles)
        if mol is not None and mol.GetNumAtoms()>0:
            return MolToSmiles(mol)


class MolGenerator(MolSampler):
    '''Molecule generator based on ULMFIT. Given a trained molecule model (e.g. LSTM), generates molecules starting from a seed text. 
    
    model_fname : Learner
        Path to a trained ULMFIT model
        
   text : str
       Seed text used to start the generative process. At the moment only an empty string is available ("")
       
       
   cpu : Bool
       If True, uses cpu to make predictions
    
    '''
    
    def __init__(self, model_fname, text='', cpu = False, **kwargs):
        super(MolGenerator, self).__init__(model_fname, text, cpu)
        self.model_fname = model_fname
        self.text = text
        self.cpu = cpu
        
    
    def generate_mols(self, max_size=100, max_mols=5, temperature=1.0):
        '''Generate molecules using a base sampler'''

        mols = filter(is_valid, [self.base_sampler(max_size=max_size,
                                                   temperature=temperature) for _ in tqdm(range(max_mols))])
        return list(mols)
    
@patch 
@delegates(MolGenerator)
def predict(m:MolGenerator, mols:List=[], canonical:bool=True, thresh:float=0.5, **kwargs):
    return NotImplemented

# @patch
# @delegates(MolSampler)
# def predict(x:MolSampler, df, smiles_column = 'Smiles', 
#             canonical=True, aug = 4, beta = 0.5,
#             thresh=0.5, **kwargs):
    
#     if len(df) < 1:
#         return df
    
    
#     if canonical:
#         df[smiles_column] = df[smiles_column].apply(lambda x : is_valid(x))
        
#     canonical_dl = x.classifier.dls.test_dl(df)
#     _,canonical_probas = x.classifier.get_preds(dl=canonical_dl)
  
    
#     aug_df = smiles_augmentation(df=df,smiles_column=smiles_column,N_rounds=aug)
#     aug_dl = x.classifier.dls.test_dl(aug_df)
#     _,aug_probas = x.classifier.get_preds(dl=aug_dl)

#     mean_probas = np.stack([aug_probas[i: i + aug, 1].mean(0) for i in range(0, aug_probas.shape[0], aug)])

#     tta_probas = torch.lerp(tensor(mean_probas), tensor(canonical_probas[:, 1]), beta)
    
#     df['probas'] = tta_probas.numpy()
#     df['preds'] = torch.where(tta_probas>=thresh,1,0)
#     return df