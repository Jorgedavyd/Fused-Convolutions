from fusedconv import FusedCovn1D, FusedCovn2D, FusedCovn3D
from typing import Callable
from ..python.typing import FusedArgs
import yaml

def createArgs(data: Dict) -> FusedArgs:
    return data

class testConfig:
    path: Callable[[int], str] = lambda x: f'./tests/config{x}.yml'
    @staticmethod
    def __call__(idx: int) -> FusedArgs | None:
        config_file: str = testConfig.path(idx)
        with open(config_file) as stream:
            try:
                data = yaml.safe_load(stream)
                return createArgs(data)
            except yaml.YAMLError as exc:
                print(exc)

