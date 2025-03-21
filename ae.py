from tensorflow.keras import Model
from tensorflow.keras.layers import Input

class Autoencoder:
    """
    Autoencoder represents a Deep Convolutional autoencoder architecture with mirrored encoder a nd decoder components.
    """
    def __init__(self,input_shape,conv_filters,conv_kernels,conv_strides,latent_space_dim):
        self.input_shape = input_shape # [28,28,1]
        self.conv_filters = conv_filters # [2,4,8]
        self.conv_kernels = conv_kernels # [3,5,3]
        self.conv_strides = conv_strides # [1,2,2]
        self.latent_space_dim = latent_space_dim
        
        self.encoder = None
        self.decoder = None
        self.model = None
        
        self.num_conv_layers = len(conv_filters)
        self.build()
        
    
    def build(self):
        self.build_encoder()
        self.build_decoder()
        self.build_autoencoder()
        
    def build_encoder(self):
        encoder_input = self.add_encoder_input()
        conv_layers = self.add_conv_layers(encoder_input)
        bottleneck = self.add_bottleneck(conv_layers)
        self.encoder = Model(encoder_input, bottleneck, name="encoder")
        
    def add_encoder_input(self):
        return Input(shape=self.input_shape, name="encoder_input")
    
    def add_conv_layer(self, encoder_input):
        """Creates all convolutionals blocks in encoder."""
        X = encoder_input
        