from keras.layers import Dense, Dropout, Flatten, Concatenate
from keras.models import Model

def model_discriminatorFlat(Inputs,nclasses,nregclasses,dropoutRate=0.5):
    """
    Discriminator to distinguish between real- and simulated data
    """

    globalvars = Inputs[0]
    cpf    =     Flatten(name='cpf_input_flatten')     (Inputs[1])
    npf    =     Flatten(name='npf_input_flatten')     (Inputs[2])
    vtx    =     Flatten(name='vtx_input_flatten')     (Inputs[3])

    x = Concatenate()([globalvars,cpf,npf,vtx])

    x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)

    predictions = Dense(1, activation='sigmoid',kernel_initializer='lecun_uniform')(x)

    model = Model(inputs=Inputs, outputs=predictions)
    return model