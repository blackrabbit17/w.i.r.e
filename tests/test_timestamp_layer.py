import torch
from wire.layers.timestamp import CyclicTimestampEncoding


def test_cyclic_encoding_output_shape():
    date = ["2005-09-06 16:32:00", ]
    layer = CyclicTimestampEncoding()
    output = layer(date)
    assert output.shape == (1, 8,)

def test_cyclic_encoding_output_values():
    date = ["2005-09-06 16:32:00", ]
    layer = CyclicTimestampEncoding()
    output = layer(date)[0]
    for i in range(0, 8, 2):
        sin_val = output[i]
        cos_val = output[i + 1]

        assert sin_val >= -1 and sin_val <= 1
        assert cos_val >= -1 and cos_val <= 1
