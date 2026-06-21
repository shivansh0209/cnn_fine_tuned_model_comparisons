1. I didnt know that there exists some crazy stuff like gradcam which compared thigs so closely its just like an xray

2. Didnt know about Mobilenet and why it is so superior

3. When fine tuning the VGG model we first only trained the head because if totally ranodmized weights gradient will be very big and may destory the base's weight that we got from imagenet.

4. As we increase the complexity with time we decrease the leanring rate very much because when we need only minor twekas in our weights and biases then the learning rate should be small. Thats why we can even use callbacks like ReduceLRrate as the model grows

5. Data augmentation on the fly using keras layers was new to me just like the rescaling layer. Keras make it so easy that everything happens on the fly and we dont need to write extra code. Didnt know the practical jump of using DA which here was about 10+%

6. Got to know that even if all the image data is in the sam efolder we can still use the image data form direct by keras to make validation and train datasest non overlapping using the subset seed and validation split keyword.

## MobileNet vs VGG16
For understanding the MobileNet superiority from VGG16 , we neeed to understand two main concepts one is DepthWise Sepearable Convulations and Inverted Residual Blocks

### Depthwise Separable Convulations
We will understand this with an example like say suppose one image input of 32, 32, 3 entered and the filter size is 3, 3 so in standard conv this filter tried to mathc the depth so it became of size 3, 3, 3 and say suppose if we needed 64 feature maps from here then there should be 64 such filters

So total parameters are 3* 3 * 3 *64 = 1728

But in depth wise one filter does not matches the depth but creates filters so that there is exactly one filter available for each depth/channel. Now this will now produce 3 feature maps of 32 X 32. Now we need 64 feature maps from these 3 feature maps. So suppose if we want to make one feature map from these three featue maps then

for making the top right pixel of that one target feature maps(xtarget) from the top right pixel of those three feature maps(x1, x2, x3) we need a linear combination of these three wich has coeefecients say w1, w2, w3. so 

xtarget = w1x1 + w2x2 + w3x3

Now this way one feature map will be made now to make 16 of them we need 16 such w combos. Hence 16 * 3

Total params = 3 * 3 * 3 + 16 * 3 = 75


Hence you can see the dip in number of params and thus behind the scenes there will be dip in number of mathematical operations as well.

### Residual Network
When the newtork grows deeper and deeper then may be due to shrinkage of weights there is some disturbance in learning but the residual network saves us from that.

If suppose at some mid point of network the state is 95 and need to reach 100 through the remaining network but the remainng nwtrok can worsen it or turn to 0 so the residual network will act as a bridge between the state of 95 and the layer just before the output activation function skipping the stale remaining network and then the residual network just need to learn to add thie difference of 5 to 95 so that we can get 100 value?

## Combining inverted residual and Depthwise in MobileNet

### Standard Residual

In a standard ResNet residual block, the network follows a compress-then-expand strategy to save on computation:

Compress: It takes a wide input (many channels), uses a $1 \times 1$ convolution to squash it down to a small number of channels.

Process: It does the heavy spatial convolution on this narrow, compressed bottleneck.

Expand: It uses another $1 \times 1$ convolution to blow it back up to a large number of channels.

Shortcut: The skip connection connects the wide inputs to the wide outputs.

Classic Flow: Wide Input $\rightarrow$ Narrow Bottleneck $\rightarrow$ Wide Output (Shortcut connects the Wide parts)

### Inverted Residual
The Inverted Residual Block (The "Narrow-Wide-Narrow" Approach)MobileNetV2 does the exact opposite. It uses an expand-then-compress strategy.Because MobileNet uses Depthwise Separable Convolutions (which we know are incredibly cheap computationally), it doesn't need to hide the spatial processing inside a narrow bottleneck. In fact, if you compress the channels before a depthwise convolution, the network loses too much valuable information.So, the inverted residual block works like this:

Expansion ($1 \times 1$ Conv): It takes a narrow input (low channels) and blows it up to a much wider state (typically $6\times$ the original channels).

Depthwise Convolution ($3 \times 3$): It performs the spatial filtering on this wide, high-dimensional state where the information is separated and easy to process.

Projection / Compression ($1 \times 1$ Conv): It squashes the wide state back down to a narrow, low-channel output.

Shortcut: The skip connection connects the narrow inputs to the narrow outputs.

Inverted Flow: Narrow Input $\rightarrow$ Wide Expansion $\rightarrow$ Narrow Output (Shortcut connects the Narrow parts)


```python
def inverted_residual_block(x, expanded_channels, output_channels):
    """
    A single MobileNetV2-style Inverted Residual Block.
    """
    # Save the input for the shortcut connection
    shortcut = x
    
    # --- STEP 1: EXPANSION (Pointwise 1x1 Conv) ---
    # Blow up the narrow input channels into a wider space
    x = layers.Conv2D(expanded_channels, (1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6.0)(x) # MobileNet uses ReLU6 to bound values
    
    # --- STEP 2: SPATIAL FILTERING (Depthwise 3x3 Conv) ---
    # Cheaply filter the spatial features channel-by-channel in the wide space
    x = layers.DepthwiseConv2D((3, 3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6.0)(x)
    
    # --- STEP 3: PROJECTION / COMPRESSION (Pointwise 1x1 Conv) ---
    # Squash the wide space back down to a narrow number of channels
    x = layers.Conv2D(output_channels, (1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    # Note: No activation function here! MobileNet uses a linear bottleneck.

    # --- STEP 4: SHORTCUT BRIDGE ---
    # Element-wise addition: F(x) + x
    # This only works if input channels match output channels perfectly.
    if shortcut.shape[-1] == output_channels:
        x = layers.add([x, shortcut])
        
    return x
```

Just put it at the end of the conv network.