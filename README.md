ALCA
==========

Adaptive approach for sparse representations using the locally competitive algorithm for audio , available at <https://github.com/SoufiyanBAHADI/ALCA>. This is the code for the preprint available at https://arxiv.org/abs/2109.14705

# Installation

The required python packages are listed in requirements.txt, and can be installed with: 
```bash
pip install -r requirements.txt
```

# Usage

```
cd ALCA
python main.py

optional arguments:
  -h, --help            shows this help message and exit
  -p PATH, --path PATH  The path of the data set.
  --tau TAU             Neurons' time constant.
  --dt DT               Euler's resolution method clock.
  --threshold THRESHOLD
                        Firing threshold.
  --stride STRIDE       Stride size.
  --ker-len KER_LEN     Kernels' length.
  --num-chan NUM_CHAN   Number of channels.
  --iters ITERS         The LCA's iterations.
  --optimizer {sgd,adam}
                        The optimizer needed for training.
  --lr LR               Learning rate.
  --batch-size BATCH_SIZE
                        the size of each mini batch.
  --buffer-size BUFFER_SIZE
                        The size of the buffer where to store steady states
                        for backpropagation through time algorithm
  -e EPOCHS, --epochs EPOCHS
                        number of epochs.
  --eval                Specifies the evaluation. If false the algorithm will
                        run in training mode
  -v, --verbose         allows the program verbosity
  --random-init         parameters are initiallized randomly
  --resume RESUME       The epoch from which the learning will resume
  --plot                If specified the program will plot all outputs. --eval
                        should be specified
```


# Acknowledgements

© Copyright (June 2021) Soufiyan Bahadi, prof. Jean Rouat, prof. Éric Plourde. University of Sherbrooke. [NEuro COmputational & Intelligent Signal Processing Research Group (NECOTIS)](http://www.gel.usherbrooke.ca/necotis/)

<img src="images/necotis.png" width="250" /> <img src="images/UdeS.jpg" width="250" />
