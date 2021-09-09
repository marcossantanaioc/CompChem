from chembl_structure_pipeline import standardizer, checker

from rdkit import Chem
from rdkit import rdBase

from rdkit.Chem.FilterCatalog import *

from rdkit.Chem import rdchem, rdmolops, SanitizeMol
from rdkit.Chem.SaltRemover import SaltRemover as saltremover
from rdkit.Chem.MolStandardize.rdMolStandardize import LargestFragmentChooser

rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.info')

from tqdm.notebook import tqdm


_saltremover = saltremover()
_unwanted = Chem.MolFromSmarts('[!#1!#6!#7!#8!#9!#15!#16!#17!#35!#53]')

class DatasetSanitizer():
    def __init__(self, mol_list):
        self.data_size = len(mol_list)
        self.mol_list = map(Chem.MolFromSmiles, mol_list)
        
    def __len__(self):
        return len(self.mol_list)
    
    def __str__(self):
        return f'Data has {len(self.mol_list)} records'
    
    
    def getlargestFragment(self, mol):
        '''Get largest fragments in a molecule'''
        frags = rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        maxmol = None
        for mol in frags:
            if mol is None:
                continue
            if maxmol is None:
                maxmol = mol
            if maxmol.GetNumHeavyAtoms() < mol.GetNumHeavyAtoms():
                maxmol = mol
        return maxmol

    def remove_unwanted(self, mol):
        '''Remove molecules with unwanted elements (check the _unwanted definition) and isotopes'''
        if not mol.HasSubstructMatch(_unwanted) and (sum([atom.GetIsotope() for atom in mol.GetAtoms()])==0):
            return mol


    def remove_salts(self, mol):
        '''Strip salts'''
        mol = _saltremover.StripMol(mol, dontRemoveEverything=True)
        mol = rdmolops.RemoveHs(mol, implicitOnly=False, updateExplicitCount=False, sanitize=True)
        mol = self.getlargestFragment(mol)

        return mol


    def normalize_mol(self, mol):
        '''Normalizer
        
        See https://github.com/chembl/ChEMBL_Structure_Pipeline/wiki/Work-done-by-each-step#standardize_molblock
        for details on the normalization steps
        
        '''
        parent, _ = standardizer.get_parent_mol(mol)
        norm_mol = standardizer.normalize_mol(parent)

        return norm_mol

    def process_mol(self, mol):       
        '''Fully process one Mol'''
                   
        # Remove salts and molecules with unwanted elements (See _unwanted definition above)
        mol = self.remove_unwanted(self.remove_salts(mol))
        
        if isinstance(mol, rdchem.Mol):
            # Normalize molecule using chembl_structure_pipeline
            mol = self.normalize_mol(mol)
            return Chem.MolToSmiles(mol)
        return None
    
    def sanitize_dataset(self):
        processed_mols = list(map(self.process_mol, tqdm(self.mol_list, total=self.data_size)))
        return processed_mols
