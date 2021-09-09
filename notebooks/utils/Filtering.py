from rdkit.Chem.FilterCatalog import *
from rdkit import Chem


def _pains():
    '''Define PAINS substructures'''
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
    catalog = FilterCatalog(params)

    return catalog

class PAINSFilter():
    '''Filter a collection of molecules using a catalog of rules as defined by RDKit    
    
    '''
    def __init__(self):
        self.catalog = _pains()

    @property
    def pains_catalog(self):
        return self.catalog
    
    
    @pains_catalog.setter 
    def pains_catalog(self, new_catalog):
        self.catalog = new_catalog
    
    def pains_filter(self, mol, catalog=None):

        '''Identify PAINS among a set of molecules

        Arguments:

        mol : rdchem.Mol or str

            RDKit mol object or SMILES representing a molecule

        Returns:
            pains_name : str
                PAINS identifier

        '''
        if not catalog:
            catalog = self.catalog
            
        assert catalog.GetNumEntries() == 480

        try:
            if not isinstance(mol, Chem.rdchem.Mol): mol = Chem.MolFromSmiles(mol)
            entry = catalog.GetFirstMatch(mol)
            if entry:
                pains_name = entry.GetDescription()
                return pains_name
        except:
            pass
