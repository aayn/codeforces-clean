# from sklearn.feature_extraction import CountVectorizer
from pathlib import Path
from torch import nn

SKAI_MPATH = Path('checkpoints/')


class MWrapper:
    "Model wrapper class."
    def __init__(self, model, name):
        """Takes as input a sklearn or pytorch model. Keeps track of 
        related information.
        
        mname: used to create a directory with checkpoints and other data.
        """
        self.model = model
        self.name_ = name
        if isinstance(model, nn.Module):
            self.modeltype_ = 'pytorch'
        else:
            self.modeltype_ = 'sklearn'
        self.path_ = SKAI_MPATH/name
        self.create_model_store()
        if self.type == 'sklearn':
            self.pipeline = model.pipeline
            self.parameters = model.parameters
    
    @property
    def type(self): return self.modeltype_
    
    @property
    def name(self): return self.name_

    @property
    def path(self): return self.path_
    
    def create_model_store(self):
        "Creates a storage location for the model."
        try:
            self.path.mkdir()
        except FileExistsError:
            print(f'Note: Model directory for {self.name} exists.')

class SKModel:
    def __init__(self, pipeline, parameters):
        self.pipeline = pipeline
        self.parameters = parameters
        