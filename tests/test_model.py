import numpy as np
import torch

from bdo import BDOEncoder, RobustScaler, regular_times


def small_model():
    torch.manual_seed(0)
    return BDOEncoder(input_dim=450, embed_dim=64, depth=2, num_heads=2).eval()


def test_shapes():
    model = small_model()
    x, t = torch.randn(3, 50, 450), torch.arange(50.0).expand(3, 50) * 0.72
    assert model(x, t).shape == (3, 50, 64)
    assert model.embed_subject(x, t).shape == (3, 64)


def test_sequences_are_independent():
    model = small_model()
    x, t = torch.randn(4, 40, 450), torch.rand(4, 40).cumsum(1)
    batched = model(x, t)
    single = torch.cat([model(x[i:i + 1], t[i:i + 1]) for i in range(4)])
    assert torch.allclose(batched, single, atol=1e-5)


def test_save_and_load(tmp_path):
    model = small_model()
    path = tmp_path / "model.pt"
    torch.save({"config": model.config, "state_dict": model.state_dict()}, path)
    loaded = BDOEncoder.from_pretrained(str(path))
    x, t = torch.randn(1, 20, 450), torch.arange(20.0)[None]
    assert torch.equal(model(x, t), loaded(x, t))


def test_robust_scaler(tmp_path):
    rng = np.random.default_rng(0)
    scans = [rng.normal(5.0, 2.0, size=(120, 450)) for _ in range(4)]
    scaler = RobustScaler().fit(scans)
    out = scaler.transform(scans[0])
    assert out.shape == (110, 450) and out.dtype == np.float32
    pooled = np.concatenate([scaler.transform(s) for s in scans])
    assert np.allclose(np.median(pooled, axis=0), 0, atol=1e-6)
    scaler.save(tmp_path / "scaler.npz")
    assert np.array_equal(RobustScaler.load(tmp_path / "scaler.npz").transform(scans[1]), scaler.transform(scans[1]))
    assert np.allclose(regular_times(3, 0.72), [0, 0.72, 1.44])
