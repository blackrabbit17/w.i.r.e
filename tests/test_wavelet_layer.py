import torch

from wire.layers.wavelet import WaveletTransform

"""

WARNING : These tests are boilerplate, the expected = torch.tensor() values are not correct.

"""

def test_WaveletTransform_init():
    wt = WaveletTransform("db4", "zero", True)
    assert wt.wavelet_type == "db4"
    assert wt.mode == "zero"
    assert wt.padding == True

    wt = WaveletTransform("haar", "zero", False)
    assert wt.wavelet_type == "haar"
    assert wt.mode == "zero"
    assert wt.padding == False

def test_WaveletTransform_forward_db4():

    x = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])

    wt = WaveletTransform("db4", "zero", True)

    result = wt(x)

    expected = torch.tensor(
        [ 7.3697e+00,  2.9609e+00,  1.0282e+01,  4.9193e+00,  4.1403e+00,
          1.2864e+01,  1.5474e+01, -4.2430e-01, -1.1344e+00,  9.9416e-01,
          3.2229e-01, -1.0294e+00,  1.6662e-02,  1.2550e+00,  2.3713e-02,
          4.0962e-02, -6.4675e-02,  9.5410e-17, -2.3713e-02, -4.0962e-02,
          6.4675e-02],
        dtype=torch.float64)

    assert torch.allclose(result, expected, rtol=1e-4, atol=1e-4)
