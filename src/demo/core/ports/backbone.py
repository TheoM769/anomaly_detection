from abc import ABC, abstractmethod

class Backbone(ABC):

    @abstractmethod
    def load_weights(self):
        pass