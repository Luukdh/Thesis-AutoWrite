Here is a short explanation on the naming convention of the models.

- "base" : The base architecture based on U-Net made to accept one dimensional input.
- "unet_": Architecture alteration based on the base architecture.
- "das_1": A architecture based on the base architecture made for experiments on the DAS4.
- "das_2": Second architecture based on the base architecture made for experiments on the DAS4.

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
- "nopad": Architecture that implements no padding during convolutional operations.

