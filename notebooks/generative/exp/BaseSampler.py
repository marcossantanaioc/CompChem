from generative_basics import *
from exp.Filtering import *

class MolSampler():
    def __init__(self,model_fname:str,
                 text:str='',max_size:int=120,
                 max_mols:int=100,temperature:float=0.7,
                 cpu:bool=False):
        
        self.model = load_learner(model_fname, cpu=cpu)
        self.text = text
        self.max_size = max_size
        self.temperature = temperature
        self.max_mols = max_mols    
    
@patch
@delegates(MolSampler)
def base_sampler(x:MolSampler, max_size=100, temperature=0.7, **kwargs):
    '''Base sampler to generate one SMILES using a chemistry model trained with fastai
    temperature : sampling temperature (default = 0.7)
    max_size : maximum size of the SMILES strings (default = 100)'''
    act = getattr(x.model.loss_func, 'activation', noop)
    x.model.model.cuda()
    x.model.model.reset()    # Reset the model
    stop_index = x.model.dls.train.vocab.index(BOS)        # Define the stop token
    idxs = x.model.dls.test_dl([x.text]).items[0].to(x.model.dls.device)
    nums = x.model.dls.train_ds.numericalize     # Numericalize (used to decode)
    accum_idxs = []                   # Store predicted tokens
    x.model.model.eval()

    for _ in range(max_size):
        with torch.no_grad(): preds = x.model.model(idxs[None])[0][-1]
        
        res = act(preds)

        if x.temperature != 1.: res.pow_(1 / temperature)
        idx = torch.multinomial(res, 1).item()
        if idx != stop_index:
            accum_idxs.append(idx)
            idxs = TensorText(idxs.new_tensor([idx]))
        else:
            break
    decoded = ''.join([nums.vocab[o] for o in accum_idxs if nums.vocab[o] not in [BOS, PAD]])  # Decode predicted tokens
    yield decoded
    
@patch
@delegates(MolSampler)
def generate_mols(x:MolSampler, max_size=100, max_mols=5, temperature=0.7,**kwargs):
    '''Generate molecules using a base sampler'''
        
    return [x.base_sampler(max_size=max_size, temperature=temperature) for i in range(max_mols)]