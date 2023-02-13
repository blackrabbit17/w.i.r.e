```
 _____ _     _                   _          _
/__   \ |__ (_)_ __   __ _ ___  | |_ ___   | |_ _ __ _   _
  / /\/ '_ \| | '_ \ / _` / __| | __/ _ \  | __| '__| | | |
 / /  | | | | | | | | (_| \__ \ | || (_) | | |_| |  | |_| |
 \/   |_| |_|_|_| |_|\__, |___/  \__\___/   \__|_|   \__, |
                     |___/                           |___/
```
Volume encoding:

    - Binning
    - Embedding

Alternate topologies:

    - Branched with deep features and merge (essentially embeddings)
      - Topology for these (FC, CONV?)
      - Num learnable parameters

    - Convolutional Networks

    - Multi-layer stacked


More advanced wavelet layer:

    - Differential will mean the coefficients can be learnt with backpropagation

    - GPU implementation with pytorch tensors

    - Support for continuous wavelet functions:
      - implemented - db4, db8, haar
      - not implemented: morlet, mexh, paul

    - Wavedec level hard coded to 1, but the dwt_max_level is computed internally via:
      > level = int(np.floor(np.log2(data_len / wavelet.dec_len + 1)))
      Which, due to our short signal length is rounding to zero, and causing the wavelet functions to return the raw input signal!
        - Investigate: Does a longer signal cause dwt_max_level to compute level > 0 ?
        - should we just add level as a hyperparameter and grid search it for small values: {1, 2, 4} ?


Forecast close > 1 time step ahead!

```
Summary so far:
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

The best performing models were the ones with:
  Unsurprising:
  - Larger hidden units (512)
  - Smaller batch size (64)
  - Higher learning rate (0.01) vs (0.001)
  - Larger dataset (20% of total)

  Surprising:
  - Univariate
```
