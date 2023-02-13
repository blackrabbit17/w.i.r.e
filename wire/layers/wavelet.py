import torch
import pywt


"""
    Common wavelets for Time Series analysis

    Daubechies wavelets: Daubechies wavelets are a family of orthogonal wavelets that are widely used in
                         signal processing, including time series analysis. They are known for their good
                         time and frequency localization properties and fast computation. Daubechies
                         wavelets of order 4 and 8 are particularly popular for time series analysis.
    pywt wavelet_type = "db1", "db2", "db3", "db4", "db5", "db6", "db7", "db8", "db9", "db10", "db11",
                        "db12", "db13", "db14", "db15"


    Morlet wavelet: The Morlet wavelet is a complex-valued wavelet that is widely used in time series
                    analysis, especially for the analysis of signals with periodic or quasi-periodic
                    behavior. The Morlet wavelet provides good time and frequency localization,
                    making it well suited for the analysis of signals with multiple frequency
                    components.
    pywt wavelet_type = "morl"


    Mexican hat wavelet: The Mexican hat wavelet is a widely used wavelet in time series analysis,
                         especially for the analysis of signals with transient behavior. The Mexican
                         hat wavelet has good time and frequency localization properties, and it
                         provides a good balance between time resolution and frequency resolution.
    pywt wavelet_type = "mexh"


    Haar wavelet: The Haar wavelet is a simple wavelet that is widely used for time series analysis,
                  especially for the analysis of signals with abrupt changes or sharp transitions.
                  The Haar wavelet provides good time resolution, but limited frequency resolution.
    pywt wavelet_type = "haar"


    Paul wavelets: Paul wavelets are a family of wavelets that are designed to have good time and
                   frequency localization properties, making them well suited for time series
                   analysis. Paul wavelets are especially useful for the analysis of signals with
                   multiple frequency components, as they provide good time resolution at low
                   frequencies and good frequency resolution at high frequencies.
    pywt wavelet_type = "paul"
"""


class WaveletTransform(torch.nn.Module):
    def __init__(self, wavelet_type="db4", mode="zero", padding=True):
        super(WaveletTransform, self).__init__()

        # Wavelet specific properties
        self.wavelet_type = wavelet_type
        self.wavelet = pywt.Wavelet(wavelet_type)
        self.filter_size = len(self.wavelet.dec_lo)

        self.mode = mode
        self.padding = padding

    def forward(self, x):
        x = x.squeeze()
        if self.padding:
            padding = (0, x.shape[-1] % 2)
            if self.mode == "zero":
                x = torch.nn.functional.pad(x, padding, mode="constant", value=0)
            else:
                x = torch.nn.functional.pad(x, padding, mode=self.mode)

        coeffs = pywt.wavedec(x.numpy(), self.wavelet_type, level=1)
        coeffs_tensor = [torch.from_numpy(c) for c in coeffs]
        return torch.cat(coeffs_tensor, dim=-1)

    def get_output_dim(self, input_size) -> int:

        # We have to execute the forward pass to get the output size
        # This is because the output size depends on the input size
        # and the padding mode

        x = torch.randn(1, 1, input_size)
        out = self.forward(x)
        return out.shape[-1]
