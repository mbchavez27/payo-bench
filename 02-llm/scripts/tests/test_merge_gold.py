"""merge_gold: combining two machines' gold/ folders when the GPU work was split by model."""

from merge_gold import check_compatible, copy_outputs

SETUP = {"precision": "fp32", "engine": {"tensor_parallel_size": 2, "cuda_graphs": "decode"}}


def test_compatible_when_models_differ_and_setup_matches():
    ours = {"gemma": {"recognition,advice-p0": SETUP, "advice-oracle-grid": SETUP}}
    theirs = {"sealion": {"recognition,advice-p0": SETUP}}
    assert check_compatible(ours, theirs) == []


def test_rejects_same_model_or_different_setup():
    ours = {"gemma": {"main": SETUP}}
    assert any("gemma is in both" in p for p in check_compatible(ours, {"gemma": {"main": SETUP}}))
    other = {"sealion": {"main": {**SETUP, "precision": "fp8"}}}
    assert any("differ" in p for p in check_compatible(ours, other))


def test_copy_outputs_skips_records_and_flags_conflicts(tmp_path):
    src, dst = tmp_path / "theirs", tmp_path / "ours"
    (src / "emotion-recognition").mkdir(parents=True)
    (src / "emotion-recognition" / "predictions_sealion.jsonl").write_text("{}\n", encoding="utf-8")
    (src / "chat_templates").mkdir()
    (src / "chat_templates" / "shared.txt").write_text("a", encoding="utf-8")
    (src / "run_record.json").write_text("{}", encoding="utf-8")
    (dst / "chat_templates").mkdir(parents=True)
    (dst / "chat_templates" / "shared.txt").write_text("b", encoding="utf-8")

    copied, conflicts = copy_outputs(src, dst)
    assert copied == 1 and (dst / "emotion-recognition" / "predictions_sealion.jsonl").exists()
    assert conflicts == ["chat_templates/shared.txt"] and not (dst / "run_record.json").exists()
