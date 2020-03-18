
import numpy as np
import tvm
from tvm import te
from tvm import relay
from tvm.relay import testing

import binary_layers

"""The params of models"""
vgg16 = {
    'conv2d': {
        # params: {input_channel, input_height, input_weight, ouptut_channel, kernel_size,
        #  stride, padding, activation_bits, weight_bits}
        'qconv_2': {'params': [64, 112, 112, 64, 3, 1, 1, 4, 4]},
    },
}

def get_bitserial_conv2d(model, layer_name, batch_size=1, dtype='int8'):
    params = model['conv2d'][layer_name]['params']
    
    image_shape = params[0], params[1], params[2]
    
    data_shape = (batch_size, ) + image_shape 
    
    output_channel = params[3]

    
    kernel_size = params[4]

    weight_shape = output_channel, params[0], kernel_size, kernel_size

    stride = params[5]
    padding = params[6]
    
    activation_bits = params[7]
    weight_bits = params[8]
    
    data = relay.var("data", shape=data_shape, dtype=dtype)
    weight = relay.var(layer_name + "_weight", shape=weight_shape, dtype=dtype)

    net = binary_layers.bitserial_conv2d(data=data, weight=weight, strides=(stride, stride),
                                        padding=(padding, padding), channels=output_channel, 
                                        kernel_size=(kernel_size, kernel_size), 
                                        activation_bits=activation_bits, weight_bits=weight_bits, 
                                        pack_dtype='uint8', name=layer_name)
    
    
    net = relay.Function(relay.analysis.free_vars(net), net)

    mod, params = testing.create_workload(net)

    # We only needs to return this three variables
    return mod, params, data_shape
    


    
    
    


if __name__ == '__main__':
    mod, params = get_bitserial_conv2d(vgg16, 'qconv_2', 1, 'int8')
    print(mod)
    print(params)
   
    