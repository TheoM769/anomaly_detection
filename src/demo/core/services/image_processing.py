from core.ports.model import Model
from core.ports.backbone import Backbone

class ImageProcessingService:
    _registry = {}  # Dictionnaire pour associer les noms des processeurs à leurs implémentations
    _instances = {}  # Dictionnaire pour stocker les instances actives
    _current_model_name = None  # Pour suivre le modèle actuellement utilisé

    @classmethod
    def register_processor(cls, name: str, adapter: tuple[Model, Backbone, str, int]):
        cls._registry[name] = adapter
        
    @classmethod
    def get_processor(cls, name: str) -> Model:
        if cls._current_model_name is not None and cls._current_model_name != name and name in cls._registry.keys():
            # Libérer le modèle précédent
            if cls._current_model_name in cls._instances.keys():
                cls._instances[cls._current_model_name].release_resources()  # Méthode à implémenter dans vos adaptateurs
                del cls._instances[cls._current_model_name]
                print(f"Released model: {cls._current_model_name}")
        
        # Créer une nouvelle instance si nécessaire
        if name not in cls._instances.keys() and name in cls._registry.keys():
            model_constructor, backbone_constructor, kwargs = cls._registry.get(name)
            backbone = backbone_constructor()
            cls._instances[name] = model_constructor(backbone, **kwargs)
            cls._current_model_name = name
            print(f"Loaded model: {name}")
        
        return cls._instances.get(name)