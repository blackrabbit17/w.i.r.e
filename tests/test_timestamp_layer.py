import torch
from datetime import datetime

from wire.layers.timestamp import CyclicEncoding

MIN_PER_DAY = 1440


def test_cyclic_encoding_with_datetime_input():
    date = "2005-09-06 16:32:00"
    layer = CyclicEncoding(MIN_PER_DAY)
    output = layer(date)
    assert isinstance(output, torch.Tensor)

"""
def test_cyclic_encoding_output_shape():
    date = datetime(2023, 2, 11, 12, 0, 0)
    layer = CyclicEncoding(MIN_PER_DAY)
    output = layer(date)
    assert output.shape == (2,)

def test_cyclic_encoding_output_values():
    date = datetime(2023, 2, 11, 12, 0, 0)
    layer = CyclicEncoding(MIN_PER_DAY)
    output = layer(date)
    sin_val, cos_val = output
    assert sin_val >= -1 and sin_val <= 1
    assert cos_val >= -1 and cos_val <= 1

def test_cyclic_encoding_with_hardcoded_datetime():
    date = datetime(2023, 2, 11, 12, 0, 0)
    layer = CyclicEncoding(MIN_PER_DAY)
    output = layer(date)  # sin_val, cos_val
    expected_output = torch.tensor([0.5, 0.8660254037844386])
    assert torch.allclose(output, expected_output, atol=1e-5)
"""