from abc import abstractmethod, ABCMeta


class ClassifierBase(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass