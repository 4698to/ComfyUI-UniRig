from typing import Dict, List, Union
from dataclasses import dataclass
import yaml
from box import Box

from .data_spec import ConfigSpec
import logging

log = logging.getLogger("unirig")
@dataclass
class OrderConfig(ConfigSpec):
    '''
    Config to handle bones re-ordering.
    '''
    
    # {skeleton_name: path}
    skeleton_path: Dict[str, str]
    
    # {cls: {part_name: [bone_name_1, bone_name_2, ...]}}
    parts: Dict[str, Dict[str, List[str]]]
    
    # {cls: parts of bones to be arranged in [part_name_1, part_name_2, ...]}
    parts_order: Dict[str, List[str]]
    
    @classmethod
    def parse(cls, config):
        cls.check_keys(config)
        skeleton_path = config.skeleton_path
        parts = {}
        parts_order = {}
        for (cls, path) in skeleton_path.items():
            assert cls not in parts, 'cls conflicts'
            d = Box(yaml.safe_load(open(path, 'r')))
            parts[cls] = d.parts
            parts_order[cls] = d.parts_order
        return OrderConfig(
            skeleton_path=skeleton_path,
            parts=parts,
            parts_order=parts_order,
        )

class Order():
    
    # {part_name: [bone_name_1, bone_name_2, ...]}
    parts: Dict[str, Dict[str, List[str]]]
    
    # parts of bones to be arranged in [part_name_1, part_name_2, ...]
    parts_order: Dict[str, List[str]]
    
    def __init__(self, config: OrderConfig):
        self.parts          = config.parts
        self.parts_order    = config.parts_order
    
    def make_names(self, cls: Union[str, None], parts: List[Union[str, None]], num_bones: int) -> List[str]:
        '''
        Get names for specified cls.
        '''
        names = []

        # Auto-infer parts from cls when parts list is empty
        # This handles cases where the model doesn't generate part tokens
        if len(parts) == 0 and cls is not None and cls in self.parts_order:
            log.info("Auto-inferring parts for cls='%s' (parts list was empty)", cls)
            parts = list(self.parts_order[cls])
            log.info("Inferred parts: %s", parts)

        for part in parts:
            if part is None: # spring
                continue
            if cls in self.parts and part in self.parts[cls]:
                part_names = self.parts[cls][part]
                log.info(f"Found {len(part_names)} bone names for cls='{cls}', part='{part}'")
                names.extend(part_names)
            else:
                log.warning("WARNING: cls='%s' or part='%s' not found in self.parts", cls, part)

        log.info(f"Total named bones: {len(names)}, num_bones: {num_bones}")
        if len(names) < num_bones:
            log.info(f"Filling {num_bones - len(names)} extra bones with generic names")
        assert len(names) <= num_bones, f"Expected {len(names)} bones for cls='{cls}', got {num_bones} bones"
        for i in range(len(names), num_bones):
            names.append(f"bone_{i}")
        return names
    
def get_order(config: OrderConfig) -> Order:
    return Order(config=config)