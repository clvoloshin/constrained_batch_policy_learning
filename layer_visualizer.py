from keras.models import Model
import matplotlib.pyplot as plt

class LayerVisualizer(object):
    def __init__(self, model):

        self.layer_outputs = [layer.output for layer in model.layers if layer.name not in ['mask', 'inp']]
        self.activation_model = Model(inputs=model.input, outputs=self.layer_outputs)
        
    def display_activation(self, datum, col_size, row_size, act_index):
        activation = self.activation_model.predict(datum)[act_index]
        activation_index=0
        fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
        for row in range(0,row_size):
            for col in range(0,col_size):
                ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
                activation_index += 1

        plt.show()