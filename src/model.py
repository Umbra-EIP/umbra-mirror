class SimpleModel:
    def __init__(self, bias: float = 0.0):
        self.bias = bias
        self.is_trained = False

    def train(self, data):
        if not data:
            raise ValueError("No data provided for training.")
        self.is_trained = True
        self.bias = sum(data) / len(data)

    def predict(self, value: float) -> float:
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")
        return value + self.bias
