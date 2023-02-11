import torch
import pywt
import numpy as np

"""
    Common wavelets for Time Series analysis
    https://www.continuummechanics.org/images/wavelets/wavelet_samples.jpg


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
        """
        :param wavelet_type: The type of wavelet to use. See pywt.wavelist() for a list of available wavelets.
        :param mode: The mode to use for the wavelet transform. See pywt.MODES.modes for a list of available modes.
        :param padding: Whether to pad the input with zeros to make the length a power of 2.
        """

        super(WaveletTransform, self).__init__()
        self.wavelet_type = wavelet_type
        self.mode = mode
        self.padding = padding

    def forward(self, x):
        if self.padding:
            x = torch.cat([x, torch.zeros((x.shape[0], self.wavelet_order-1), dtype=x.dtype)], dim=1)
        x = x.detach().numpy()
        result = []
        for i in range(x.shape[0]):
            cA, cD = pywt.dwt(x[i], self.wavelet_type, self.mode)
            result.append(np.concatenate([cA, cD]))
        result = torch.from_numpy(np.array(result))
        return result
