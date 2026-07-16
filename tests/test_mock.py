import tempfile
from pathlib import Path
from toolgeo.runner import run

def test_mock_end_to_end():
    with tempfile.TemporaryDirectory() as directory:
        result = run({"run":{"id":"test","seed":1,"output_dir":str(Path(directory)/"out")}, "data":{"source":"mock","n_tools":12,"n_decisions":100,"n_traces":30}, "features":{"backend":"mock","dimension":16}, "analysis":{"heldout_fraction":.25}})
        assert result["n_tools"] == 12
        assert set(result["heldout_pairwise_spearman"]) == {"confusion", "cooccurrence", "order", "substitutability"}
