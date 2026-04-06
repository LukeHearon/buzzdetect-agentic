import importlib.util
import json
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path

from src import config as cfg
from src.inference.embedding import BaseEmbedder
from src.inference.embedding import load_embedder


class BaseModel(ABC):
    """Abstract base class for all buzzdetect models"""

    # Class attributes that each embedder should define
    modelname: str = None
    embeddername: str = None
    digits_results: int = None  # how many digits should result files be rounded to?
    dtype_in: str = None

    def __init__(self, framehop_prop):
        """Initialize model
        """
        self.model = None
        self.embedder: BaseEmbedder = load_embedder(embeddername=self.embeddername, framehop_prop=framehop_prop, initialize=False)

        with open(os.path.join(cfg.DIR_MODELS, self.modelname, 'config_model.json'), 'r') as f:
            self.config = json.load(f)

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def predict(self, audiosamples):
        """Generate results for audio data"""
        pass


def load_model(modelname: str, framehop_prop: float, initialize: bool):
    """
    Generic function to load any model by name.

    Each model directory should contain:
    - model.py: Implementation of BaseModel with class attributes
    """
    model_path = Path(cfg.DIR_MODELS) / modelname

    if not model_path.exists():
        raise ValueError(f"model '{modelname}' not found in {cfg.DIR_MODELS}")

    # Import the model module, caching in sys.modules so that repeated calls
    # within the same process reuse the same class object. This is required for
    # the model's _xla_fn_cache (and any other class-level state) to work
    # across multiple load_model() calls in a single process.
    module_key = f"_buzzdetect_model_{modelname}"
    if module_key in sys.modules:
        module = sys.modules[module_key]
    else:
        spec = importlib.util.spec_from_file_location(
            module_key,
            model_path / "model.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[module_key] = module

    # Find the model class (should inherit from Basemodel)
    model_class = None
    for item_name in dir(module):
        item = getattr(module, item_name)
        if (isinstance(item, type) and
                issubclass(item, BaseModel) and
                item is not BaseModel):
            model_class = item
            break

    if model_class is None:
        raise ValueError(f"No BaseModel subclass found in {modelname}/model.py")

    # Instantiate and load
    model = model_class(framehop_prop=framehop_prop)

    if initialize:
        model.initialize()

    return model
