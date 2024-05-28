"""
Thin wrapper around the RDKit fingerprints generators.
Defines functions to genertae RDKit fingerprints from SMILES strings.
"""

from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from rdkit.Chem import MACCSkeys


def smi_to_mol(smi: str, add_hydrogens: bool = True) -> Chem.Mol:
    """Convert SMILES string to RDKit Mol object."""
    mol = Chem.MolFromSmiles(smi)
    if add_hydrogens:
        mol = Chem.AddHs(mol)
    return mol


def morgan_fp(smi: str, radius: int = 2, n_bits: int = 2048, add_hydrogens: bool = True) -> np.array:
    """Generate Morgan fingerprints."""
    mol = smi_to_mol(smi, add_hydrogens=add_hydrogens)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)

def topological_torsion_fp(smi: str, add_hydrogens: bool = True,
                        n_bits: int = 2048, target_size: int = 4,
                        chirals: bool = False) -> np.array:
    """Generate topological torsion fingerprints."""
    mol = smi_to_mol(smi, add_hydrogens=add_hydrogens)
    fp = AllChem.GetHashedTopologicalTorsionFingerprint(mol, targetSize=target_size,
                                                        nBits=n_bits,
                                                        includeChirality=chirals)
    fp_list = [int(fp[i] > 0) for i in range(n_bits)]
    return np.array(fp_list)


def pair_fp(smi: str, add_hydrogens: bool = True, n_bits: int = 2048) -> np.array:
    """Generate pair fingerprints."""
    mol = smi_to_mol(smi, add_hydrogens=add_hydrogens)
    fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
    return np.array(fp)


def macc_fp(smi: str, add_hydrogens: bool = True) -> np.array:
    """Generate MACCS fingerprints."""
    mol = smi_to_mol(smi, add_hydrogens=add_hydrogens)
    fp = MACCSkeys.GenMACCSKeys(mol)
    return np.array(fp)
