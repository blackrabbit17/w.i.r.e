
Volume encoding:
    - Binning
    - Embedding

Timestamp encoding:
    - Cyclic encoding
    - Embedding layer

Alternate topologies:
    - Branched with deep features and merge (essentially embeddings)
    - Convolutional

More advanced wavelet layer:
    - Differential will mean the coefficients can be learnt with backpropagation
    - GPU implementation with pytorch tensors
    - Support for continuous wavelet functions:
        - implemented - db4, db8, haar
        - not implemented: morlet, mexh, paul


Forecast > 1 time step ahead!
