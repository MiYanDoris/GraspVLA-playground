import os
import re
import numpy as np
from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import array_to_string
from libero.libero.envs.base_object import register_object


class CustomObjects(MujocoXMLObject):
    def __init__(self, name, obj_name):
        super().__init__(
            f"assets/playground_assets/{obj_name}/{obj_name}.xml",
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()
        self.rotation = (np.pi / 2, np.pi / 2)
        self.rotation_axis = "x"

        self.object_properties = {"vis_site_names": {}}


# List of obj_name values for automatic class generation
OBJ_NAMES = os.listdir('assets/playground_assets')

def extract_object_type(obj_name):
    """Extract the object type from obj_name (everything before the last underscore)"""
    return "_".join(obj_name.split("_")[:-1])


def to_camel_case(snake_str):
    """Convert snake_case to CamelCase"""
    components = snake_str.split('_')
    return ''.join(word.capitalize() for word in components)


def create_objaverse_class(obj_name):
    """Dynamically create an Objaverse class for the given obj_name"""
    object_type = extract_object_type(obj_name)
    camel_case_name = to_camel_case(object_type)
    class_name = f"Objaverse{camel_case_name}"
    default_name = f"objaverse_{object_type}"
    
    def __init__(self, name=default_name, obj_name=obj_name):
        super(self.__class__, self).__init__(name, obj_name)
        self.rotation_axis = "z"
    
    # Create the class dynamically
    cls = type(class_name, (CustomObjects,), {
        '__init__': __init__,
        '__module__': __name__
    })
    
    # Register the class
    register_object(cls)
    
    # Add to global namespace so it can be imported
    globals()[class_name] = cls
    
    return cls


# Automatically create and register all classes
for obj_name in OBJ_NAMES:
    create_objaverse_class(obj_name)