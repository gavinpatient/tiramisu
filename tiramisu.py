from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Dense, Input, concatenate, Activation, Dropout, BatchNormalization

def layer(in_layer,filters):
    bn=BatchNormalization()(in_layer)
    relu=Activation('relu')(bn)
    conv=Conv2D(filters, kernel_size=(3,3), padding='same')(relu)
    drop=Dropout(0.2)(conv)
    return drop

def transition_down(in_layer):
    filters=int(in_layer.shape[-1])
    bn=BatchNormalization()(in_layer)
    relu=Activation('relu')(bn)
    conv=Conv2D(filters, kernel_size=(1,1), padding='same')(relu)
    drop=Dropout(0.2)(conv)
    pool=MaxPooling2D(pool_size=(2,2))(drop)
    return pool

def transition_up(in_layer):
    filters=int(in_layer.shape[-1])
    updoot=Conv2DTranspose(filters, (3,3), strides=2, padding='same')(in_layer)
    return updoot

def dense_block(in_layer,filters,num_layers=4):
    layer_output=[in_layer, layer(in_layer, filters)]
    for _ in range(1, num_layers):
        concat=concatenate(layer_output[-2:], axis=-1)
        out=layer(concat, filters)
        layer_output.append(out)
    concat=concatenate(layer_output[1:], axis=-1)
    return concat

def get_layers(in_layer, growth_size, depth, dense_layers):
    if depth==1:
        return dense_block(in_layer,growth_size, dense_layers[0])
    densedown=dense_block(in_layer, growth_size, dense_layers[0])
    catdown=concatenate([in_layer, densedown], axis=-1)
    td=transition_down(catdown)
    deep=get_layers(td, growth_size, depth-1, dense_layers[1:])
    tu=transition_up(deep)
    catup=concatenate([tu, catdown], axis=-1)
    denseup=dense_block(catup, growth_size, dense_layers[0])
    return denseup

def Tiramisu(in_shape=(224,224,3), num_classes=32, depth=6, growth_size=16, dense_layers=[4,5,7,10,12,15]):
    filters=growth_size*3
    if type(dense_layers) != list:
        dense_layers = [dense_layers for _ in range(depth)]
    inputs = Input(in_shape)
    conv_in=Conv2D(filters,(3,3), activation='relu', padding='same')(inputs)
    
    tiramisu=get_layers(conv_in, growth_size, depth,dense_layers)
    
    conv_out=Conv2D(num_classes,(1,1), activation='softmax', padding='same')(tiramisu)
    return Model(inputs=[inputs], outputs=[conv_out])