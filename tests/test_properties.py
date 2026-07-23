from rdkit.Chem.rdChemReactions import ReactionToSmarts
from rdkit.Chem.rdmolfiles import MolFromSmiles
from syrupy.assertion import SnapshotAssertion

from gloryxr.models import MultiFAME3RModelProvider
from gloryxr.prediction import GLORYxR
from gloryxr.reactions import Reactor

PARACETAMOL = MolFromSmiles("CC(=O)Nc1ccc(O)cc1")


def test_predictions_as_expected(snapshot: SnapshotAssertion):
    gloryxr = GLORYxR(
        reactor=Reactor.load_builtin(phase="1+2"),
        models=MultiFAME3RModelProvider("models"),
    )
    reactions = gloryxr.predict_one(PARACETAMOL)

    assert [
        (ReactionToSmarts(rxn), rxn.GetDoubleProp("Score")) for rxn in reactions
    ] == snapshot
