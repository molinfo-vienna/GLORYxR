from rdkit.Chem import AllChem as Chem  # type:ignore
from syrupy.assertion import SnapshotAssertion

from gloryxr.models import LocalFAME3RModelProvider
from gloryxr.prediction import GLORYxR

PARACETAMOL = Chem.MolFromSmiles("CC(=O)Nc1ccc(O)cc1")


def test_predictions_as_expected(snapshot: SnapshotAssertion):
    gloryxr = GLORYxR(phase="1+2", models=LocalFAME3RModelProvider("models"))
    predictions = gloryxr.predict_one(PARACETAMOL)

    assert [
        (Chem.ReactionToSmarts(p.concrete_reaction), p.score) for p in predictions
    ] == snapshot
