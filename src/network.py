class NeuralNetwork:
    def __init__(self, loss_function):
        self.layers = []
        self.loss_function = loss_function

    def train(self, data, expected_values, epochs):
        for i in epochs:
            print(f"epoch: {i}")
            output = self.predict(data)
            loss = self.loss_function(expected_values, output)
            print(f"Loss: {loss}")
            # TODO: backwards pass

    def predict(self, data):
        output = data
        for layer in self.layers:
            output = layer.forward(output)
        return output
