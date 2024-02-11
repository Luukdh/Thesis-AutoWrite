Here is a short explanation on the naming convention of the plots.

- "loss" : Plots the calculated average loss over an epoch during the entire training run.
- "base" : The base architecture based on U-Net made to accept one dimensional input.
- "das"  : A architecture based on the base architecture made for experiments on the DAS4.
- "test" : Plots made during experimentation with no documented context and therefore no value.

- "minp" : "Min Max-Pooling layer", removes a Pooling layer from the base architecture.
- "plusp": "Plus Max-Pooling layer", adds a Pooling layers from the base architecture.
- "xp"   : "Extra padding", increases the kernel size by one.
- "xxp"  : "Extra Extra padding", increases the kernel size by two.
- "rc"   : "Remove convolution", removes a convolutional layer from all convolution-pairs.
- "xc"   : "Extra convolution", adds a convolutional layer to all convolution-pairs.
- "xxc"  : "Extra Extra convolution", adds two convolutional layers to all pairs.
- "lch"  : "Less channels", halves the channel count in the convolutional layers.
- "llch" : "Lot Less channels", quarters the channel count in the layers.
- "slch" : "Slightly less channels", factors the channel count by 75% (3/4 of base).

- "kern" : Main plot comparing the loss during kernel size experiments.
- "pool" : Main plot comparing the loss during pooling layer count experiments.
- "chct" : "Channel count", Main plot comparing the loss during channel count experiments.
- "comb" : Main plot comparing the loss when combining alterations.
- "conv" : Main plot comparing the loss during convolutional layer count experiments.
