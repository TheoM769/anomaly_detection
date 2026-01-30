# registry.py
import inspect

class Registry:
    def __init__(self, name):
        self._name = name
        self._model_registry = {}
        self._backbone_registry = {}
        self._instances = {}  # Dictionnaire pour stocker les instances actives
        self._current_model_name = None  # Pour suivre le modèle actuellement utilisé

    def register_model(self, name=None):
        def decorator(obj):
            key = name or obj.__name__
            if key in self._model_registry:
                raise KeyError(f"{key} already registered in {self._name}")
            self._model_registry[key] = obj
            return obj
        return decorator
    
    def register_backbone(self, name=None):
        def decorator(obj):
            key = name or obj.__name__
            if key in self._backbone_registry:
                raise KeyError(f"{key} already registered in {self._name}")
            self._backbone_registry[key] = obj
            return obj
        return decorator

    def get_model(self, name):
        if name not in self._model_registry:
            raise KeyError(f"{name} is not registered in {self._name}")
        return self._model_registry[name]
    
    def get_backbone(self, name):
        if name not in self._backbone_registry:
            raise KeyError(f"{name} is not registered in {self._name}")
        return self._backbone_registry[name]

    def _get_nested_value(self, cfg, *keys):
        current = cfg
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current
    
    def _acquire_model_lock(self, model_name):
        if self._current_model_name is not None and self._current_model_name != model_name:
                if self._current_model_name in self._instances.keys():
                    self._instances[self._current_model_name].release_resources()
                    del self._instances[self._current_model_name]
                print(f"Released model: {self._current_model_name}")
    
    def _create_model_instance(self, model_constructor, backbone_instance=None, model_params=None):
        """
        Create a model instance handling different constructor signatures gracefully.
        
        Args:
            model_constructor: The model class constructor
            backbone_instance: Optional backbone instance
            model_params: Optional model parameters dict
            
        Returns:
            Model instance
        """
        if model_params is None:
            model_params = {}
            
        # Get constructor signature
        sig = inspect.signature(model_constructor)
        param_names = list(sig.parameters.keys())
        
        # Prepare arguments based on constructor signature
        kwargs = {}
        args = []
        
        # If backbone is provided, determine how to pass it
        if backbone_instance is not None:
            # Check if constructor expects backbone as first positional argument
            if len(param_names) > 0 and param_names[0] not in ['self', 'cls']:
                args.append(backbone_instance)
            # Or as a named parameter
            elif 'backbone' in param_names:
                kwargs['backbone'] = backbone_instance
        
        # Add model parameters, filtering out those not accepted by constructor
        for key, value in model_params.items():
            if key in param_names:
                kwargs[key] = value
        
        try:
            return model_constructor(*args, **kwargs)
        except TypeError as e:
            # Fallback: try with just the backbone if provided
            if backbone_instance is not None:
                try:
                    return model_constructor(backbone_instance)
                except TypeError:
                    # Last resort: just the constructor with no args
                    return model_constructor()
            else:
                return model_constructor()

    def build_model(self, cfg):
        try:
            model_cfg = cfg["model"]
        except KeyError:
            raise KeyError("'model' is required in configuration")
        
        model_constructor = self.get_model(model_cfg["model_name"])
        if "backbone" in cfg:
            backbone_cfg = cfg["backbone"]
            backbone_constructor = self.get_backbone(backbone_cfg["backbone_name"])
            model_name = model_cfg["model_name"] + "_" + backbone_cfg["backbone_name"]
            self._acquire_model_lock(model_name)
            
            # Create backbone instance
            backbone_instance = backbone_constructor(**backbone_cfg.get("params", {}))
            
            # Create model instance with backbone
            self._instances[model_name] = self._create_model_instance(
                model_constructor, 
                backbone_instance, 
                model_cfg.get("params", {})
            )
        
        else:
            model_name = model_cfg["model_name"]
            self._acquire_model_lock(model_name)
            self._instances[model_name] = self._create_model_instance(
                model_constructor, 
                model_params=model_cfg.get("params", {})
            )
        
        self._current_model_name = model_name
        return self._instances[model_name]
    
    def build_backbone(self, cfg):
        try:
            backbone_cfg = cfg["backbone"]
        except KeyError:
            raise KeyError("'backbone' is required in configuration")
        
        backbone_name = backbone_cfg["backbone_name"]
        backbone_constructor = self.get_backbone(backbone_name)
        self._acquire_model_lock(backbone_name)
        self._instances[backbone_name] = backbone_constructor(**backbone_cfg["params"])
        self._current_model_name = backbone_name
        return self._instances[backbone_name]
