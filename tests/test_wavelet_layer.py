import torch
import numpy as np
import pytest
from WaveletTransform import WaveletTransform

def test_WaveletTransform_init():
    wt = WaveletTransform("db4", "zero", True)
    assert wt.wavelet_type == "db4"
    assert wt.mode == "zero"
    assert wt.padding == True

    wt = WaveletTransform("morl", "constant", False)
    assert wt.wavelet_type == "morl"
    assert wt.mode == "constant"
    assert wt.padding == False

def test_WaveletTransform_forward_db4():
    x = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    wt = WaveletTransform("db4", "zero", True)
    result = wt(x)
    expected = torch.tensor([[3.3241, 1.1547, -0.5236, -1.3068, -1.3068, -0.5236, 1.1547, 3.3241, 1.1547, -0.5236, -1.3068, -1.3068, -0.5236, 1.1547, 3.3241]])
    assert torch.allclose(result, expected, rtol=1e-4, atol=1e-4)

def test_WaveletTransform_forward_morl():
    x = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    wt = WaveletTransform("morl", "zero", True)
    result = wt(x)
    expected = torch.tensor([[3.3241, 1.1547, -0.5236, -1.3068, -1.3068, -0.5236, 1.1547, 3.3241, 1.1547, -0.5236, -1.3068, -1.3068, -0.5236, 1.1547, 3.3241]])
    assert torch.allclose(result, expected, rtol=1e-4, atol=1e-4)

def test_WaveletTransform_forward_mexh():
    x = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    wt = WaveletTransform("mexh", "zero", True)
    result = wt(x)
    expected = torch.tensor([[3.3241, 1.1547, -0.5236, -1.3068, -1.3068, -0.5236, 1.1547, 3.3241, 1.1547, -0.5236, -1.3068, -1.3068, -0.5236, 1.1547, 3.3241]])
    assert torch.allclose(result, expected, rtol=1e-4, atol=1e-4)

def test_WaveletTransform_forward_haar():
    x = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    wt = WaveletTransform("haar", "zero", True)
    result = wt(x)
    expected = torch.tensor([[4.0000, 2.8284, 1.4142, -0.0000, -1.4142, -2.8284, -4.0000, 0.0000, 2.8284, 1.4142, -0.0000, -1.4142, -2.8284, -4.0000, 0.0000]])
    assert torch.allclose(result, expected, rtol=1e-4, atol=1e-4)

def test_WaveletTransform_forward_paul():
    x = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    wt = WaveletTransform("paul", "zero", True)
    result = wt(x)
    expected = torch.tensor([[3.3241, 1.1547, -0.5236, -1.3068, -1.3068, -0.5236, 1.1547, 3.3241, 1.1547, -0.5236, -1.3068, -1.3068, -0.5236, 1.1547, 3.3241]])
    assert torch.allclose(result, expected, rtol=1e-4, atol=1e-4)

def test_WaveletTransform_forward_with_padding():
    x = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    wt = WaveletTransform("db4", "zero", True)
    result = wt(x)
    expected = torch.tensor([[3.3241, 1.1547, -0.5236, -1.3068, -1.3068, -0.5236, 1.1547, 3.3241, 1.1547, -0.5236, -1.3068, -1.3068, -0.5236, 1.1547, 3.3241]])
    assert torch.allclose(result, expected, rtol=1e-4, atol=1e-4)

def test_WaveletTransform_forward_without_padding():
    x = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    wt = WaveletTransform("db4", "zero", False)
    result = wt(x)
    expected = torch.tensor([[3.3241, 1.1547, -0.5236, -1.3068, -1.3068, -0.5236, 1.1547, 3.3241]])
    assert torch.allclose(result, expected, rtol=1e-4, atol=1e-4)
