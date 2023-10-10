
class NeuralNetwork:
    def __init__(self, loss_function):
        self.layers = []
        self.loss_function = loss_function

    def train(self, data, labels, epochs):
        training_data = data
        for i in epochs:
            print(f"epoch: {i}")
            for layer in self.layers:
                training_data = layer.forward(training_data)
            output = training_data
            loss = self.loss_function(output,labels)
            print(f"Loss: {loss}")
            #TODO: backwards pass
            
                



    
    def predict(self, data):
