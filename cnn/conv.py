import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d

import numpy
import pylab
from PIL import Image

rng = numpy.random.RandomState(23455)

input = T.tensor4(name='input')

w_shp = (2, 3, 9, 9)
w_bound = numpy.sqrt(3 * 9 * 9)
W = theano.shared( numpy.asarray(
            rng.uniform(
                low=-1.0 / w_bound,
                high=1.0 / w_bound,
                size=w_shp),
            dtype=input.dtype), name ='W')

b_shp = (2,)
b = theano.shared(numpy.asarray(
            rng.uniform(low=-.5, high=.5, size=b_shp),
            dtype=input.dtype), name ='b')

conv_out = conv2d(input, W)


output = T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x'))

f = theano.function([input], output)


# open random image of dimensions 639x516
img = Image.open(open('3wolfmoon.jpg'))
# dimensions are (height, width, channel)

img = numpy.asarray(img, dtype='float32') / 256.

# put image in 4D tensor of shape (1, 3, height, width)
img_ = img.transpose(2, 0, 1).reshape(1, 3, 639, 516)
filtered_img = f(img_)

# plot original image and first and second components of output
pylab.subplot(1, 3, 1); pylab.axis('off'); pylab.imshow(img)
pylab.gray();
# recall that the convOp output (filtered image) is actually a "minibatch",
# of size 1 here, so we take index 0 in the first dimension:
pylab.subplot(1, 3, 2); pylab.axis('off'); pylab.imshow(filtered_img[0, 0, :, :])
pylab.subplot(1, 3, 3); pylab.axis('off'); pylab.imshow(filtered_img[0, 1, :, :])
pylab.show()



