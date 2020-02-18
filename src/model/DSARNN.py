from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class DSARNN(Sequential):
    """
    A Dynamic Spatial Attention Recurrent Neural Network (DSARNN)
    """
    
    def __init__(self, n_features):
        """
        Create a DSARNN
        @param n_features the number of features
        """
        super().__init__()
        self.add(Dense(n_features))
