
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: dev_nb/SanitizerClass.ipynb

import pandas as pd
import numpy as np
from rdkit.Chem import rdchem, MolFromSmiles, MolToSmiles, MolFromMolBlock, MolToMolBlock, MolFromMolFile

from chembl_structure_pipeline import standardizer
from chembl_structure_pipeline import checker

from functools import partial
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.info')

# Zn not in the list as we have some Zn containing compounds in ChEMBL
# most of them are simple salts
METAL_LIST = [
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Ga", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Cd", "In", "Sn", "La", "Hf", "Ta",
    "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "Ac",
    "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm",
    "Yb", "Lu", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es",
    "Fm", "Md", "No", "Lr", "Ge", "Sb",
]


def exclude_flag(mol, remove_big=True, max_heavy_atoms=50, includeRDKitSanitization=True):
    """
    Rules to exclude structures.
    - Metallic or non metallic with more than 7 boron atoms will be excluded
      due to problems when depicting borane compounds.
    """
    rdkit_fails = False
    exclude = False
    metallic = False
    boron_count = 0

    if type(mol) == str:
        mol = Chem.MolFromMolBlock(mol, sanitize=False)
        if includeRDKitSanitization:
            try:
                Chem.SanitizeMol(mol)
            except:
                rdkit_fails = True

    for atom in mol.GetAtoms():
        a_type = atom.GetSymbol()
        if a_type in METAL_LIST:
            metallic = True
        if a_type == "B":
            boron_count += 1

    if metallic or (not metallic and boron_count >= 1) or rdkit_fails:
        exclude = True

    if remove_big:
        if mol.GetNumHeavyAtoms() >= max_heavy_atoms:
            exclude = True

    return exclude

class ChEMBLSanitizer():
    '''Performs sanitization of molecules in a dataframe using the chembl_structure_pipeline
     (https://github.com/chembl/ChEMBL_Structure_Pipeline)


     Attributes:

     dataframe : pd.DataFrame
         A pandas DataFrame with bioactivity data from ChEMBL

    smiles_column : str
        SMILES column to sanitize

     '''

    def __init__(self, dataframe, smiles_column):
        self.data = dataframe
        self.smiles_column = smiles_column
        #self.molblocks = self.multi_molblock(dataframe[smiles_column].tolist())

    def __str__(self):
        return f'Data has {len(self.dataframe)} records'

    def __len__(self):
        return len(self.dataframe)


    def add_mol_column(self, data):
        '''Add rdchem.Mol column to data'''
        return data.apply(lambda x : MolFromSmiles(x))


    def multi_molblock(self, mol_list):
        '''Converts a list of Mol objects into MolBlocks'''

        molblocks = []
        invalid_mols = []

        for struct in mol_list:
            if isinstance(struct, rdchem.Mol):
                try:
                    molblocks.append(MolToMolBlock(struct))
                except:
                    invalid_mols.append(struct)
                    continue
            else:
                try:
                    mol = MolFromSmiles(struct)
                    molblocks.append(MolToMolBlock(mol))

                except:
                    invalid_mols.append(struct)
                    continue
        return tuple(zip(mol_list, molblocks)), invalid_mols

    def molblocktosmiles(self, molblock):
        '''Convert a MolBlock object to SMILES'''
        return MolToSmiles(MolFromMolBlock(molblock))

    def get_parent(self, mol_blocks):
        '''Generate the parent compound for each compound in the dataset'''
        parents = []
        for mol, molblock in mol_blocks:
            parent_molblock, _ = standardizer.get_parent_molblock(molblock)
            parents.append((mol, self.molblocktosmiles(parent_molblock)))

        return parents

    def standardize_mols(self, mol_blocks):
        '''Generate the parent compound for each compound in the dataset'''
        standardized_mols = []
        for mol, molblock in mol_blocks:

            standardized_molblock = standardizer.standardize_molblock(molblock)
            standardized_mols.append((mol, self.molblocktosmiles(standardized_molblock)))

        return standardized_mols


    def preprocess_data(self, **kwargs):
        '''Preprocess data to remove invalid elements (See the exclude_flag function)
        '''
        max_heavy_atoms = kwargs.get('max_heavy_atoms')
        remove_big = kwargs.get('remove_big')

        # Add Mol column
        self.data['mol'] = self.add_mol_column(self.data[self.smiles_column])

        # Flag molecules for exclusion
        self.data['to_exclude'] = self.data['mol'].apply(partial(exclude_flag,
                                                                 remove_big=remove_big,
                                                                 max_heavy_atoms=max_heavy_atoms))

        data_trimmed = self.data[self.data['to_exclude']==False].reset_index(drop=True)

        return data_trimmed

    def sanitize_data(self, max_heavy_atoms=50, remove_big = True):


        # Trim data
        data_trimmed = self.preprocess_data(remove_big=remove_big,
                                            max_heavy_atoms=max_heavy_atoms)

        # Generate MolBlocks
        mol_blocks, invalid_mols = self.multi_molblock(data_trimmed[self.smiles_column])

        if len(invalid_mols) > 0:
            data_trimmed = data_trimmed.loc[~data_trimmed[self.smiles_column].isin(invalid_mols)]


        # Get parent compound
        parents = self.get_parent(mol_blocks)
        parents_df = pd.DataFrame(parents,columns=[self.smiles_column,'ParentSmiles'])

        parents_only_data = pd.concat([data_trimmed,parents_df['ParentSmiles']],join='outer',axis=1)

        # Standardize molecules
        mol_blocks, _ = self.multi_molblock(parents_only_data['ParentSmiles'])

        standardized_mols = self.standardize_mols(mol_blocks)
        standardized_df = pd.DataFrame(standardized_mols,columns=[self.smiles_column,'StandardizedSmiles'])


        processed_data = pd.concat([parents_only_data,standardized_df['StandardizedSmiles']],join='outer',axis=1)

        # Remove null values
        processed_data.dropna(subset=['mol','StandardizedSmiles'],inplace=True)

        return processed_data