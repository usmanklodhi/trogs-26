import torch

from trogs.models.classifier import build_resnet18, load_classifier
from trogs.models.crnn import GreekCRNN, Vocab, greedy_decode


def test_crnn_output_shape():
    model = GreekCRNN(num_chars=10).eval()
    out = model(torch.zeros(2, 1, 32, 100))
    assert out.shape == (2, 100 // 4 - 1, 11)  # width / 4, minus 1 from the final 2x2 conv


def test_vocab_reserves_blank():
    vocab = Vocab.from_entries([("a.jpg", "ΒΑ*"), ("b.jpg", "Α")])
    assert vocab.chars == ["*", "Α", "Β"]
    assert 0 not in vocab.idx2char
    assert vocab.encode("ΑΒ") == [2, 3]


def test_greedy_decode_collapses_repeats_and_blanks():
    idx2char = {1: "Α", 2: "Β"}
    seq = [1, 1, 0, 1, 2, 2, 0]
    scores = torch.nn.functional.one_hot(torch.tensor([seq]), num_classes=3).float()
    assert greedy_decode(scores, idx2char) == ["ΑΑΒ"]


def test_load_classifier_detects_head(tmp_path):
    for dropout in (None, 0.4):
        model = build_resnet18(5, dropout=dropout)
        path = tmp_path / f"{dropout}.pth"
        torch.save(model.state_dict(), path)
        loaded = load_classifier(str(path), 5, "cpu")
        x = torch.randn(1, 3, 64, 64)
        assert torch.equal(model.eval()(x), loaded(x))
