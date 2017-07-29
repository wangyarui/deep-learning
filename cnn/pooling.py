from theano.tensor.nnet import conv  
import theano.tensor as T  
import numpy, theano
  
  
rng = numpy.random.RandomState(23455)  
  
# symbol variable  
input = T.tensor4(name = 'input')  
  
# initial weights  
w_shape = (2,3,9,9) #2 convolutional filters, 3 channels, filter shape: 9*9  
w_bound = numpy.sqrt(3*9*9)  
W = theano.shared(numpy.asarray(rng.uniform(low = -1.0/w_bound, high = 1.0/w_bound,size = w_shape),  
                                dtype = input.dtype),name = 'W')  
  
b_shape = (2,)  
b = theano.shared(numpy.asarray(rng.uniform(low = -.5, high = .5, size = b_shape),  
                                dtype = input.dtype),name = 'b')  
                                  
conv_out = conv.conv2d(input,W)  
  
output = T.nnet.sigmoid(conv_out + b.dimshuffle('x',0,'x','x'))  
f = theano.function([input],output)  
  
   
# demo  
import pylab  
from PIL import Image  
from matplotlib.pyplot import *  
  
#open random image  
img = Image.open(open('3wolfmoon.jpg'))  
width,height = img.size  
img = numpy.asarray(img, dtype = 'float32')/256. # (height, width, 3)  
  
  
# put image in 4D tensor of shape (1,3,height,width)  
img_rgb = img.swapaxes(0,2).swapaxes(1,2) #(3,height,width)  
minibatch_img = img_rgb.reshape(1,3,height,width)  
filtered_img = f(minibatch_img)  
  
  
# plot original image and two convoluted results  
pylab.figure(1)  
pylab.subplot(1,3,1);pylab.axis('off');  
pylab.imshow(img)  
title('origin image')  
  
pylab.gray()  
pylab.subplot(2,3,2); pylab.axis("off")  
pylab.imshow(filtered_img[0,0,:,:]) #0:minibatch_index; 0:1-st filter  
title('convolution 1')  
  
pylab.subplot(2,3,3); pylab.axis("off")  
pylab.imshow(filtered_img[0,1,:,:]) #0:minibatch_index; 1:1-st filter  
title('convolution 2')  
  
#pylab.show()  
  
  
  
# maxpooling  
from theano.tensor.signal import pool  
  
input = T.tensor4('input')  
maxpool_shape = (2,2)  
pooled_img = pool.pool_2d(input,maxpool_shape,ignore_border = False)  
  
maxpool = theano.function(inputs = [input],  
                          outputs = [pooled_img])  
  
pooled_res = numpy.squeeze(maxpool(filtered_img))                
#pylab.figure(2)  
pylab.subplot(235);pylab.axis('off');  
pylab.imshow(pooled_res[0,:,:])  
title('pooling 1')  
  
pylab.subplot(236);pylab.axis('off');  
pylab.imshow(pooled_res[1,:,:])  
title('pooling 2')  
  
pylab.show()  