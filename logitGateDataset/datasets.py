from abc import *

from torch import Tensor
from torch.utils.data import Dataset


class LogitGate(Dataset, metaclass=ABCMeta):
    __constants__ = ('dataset_size, input_size')
    dataset_size: int
    input_size: int
    logit_x: tuple
    logit_y: tuple

    def __init__(self, dataset_size: int, input_size: int=2) -> None:
        self.dataset_size = dataset_size
        self.input_size = input_size
        self.logit_x, self.logit_y = self.__get_logit_table(input_size)

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        table_idx = index % (2 ** self.input_size)
        return Tensor(self.logit_x[table_idx]), Tensor(self.logit_y[table_idx])
    
    def __get_logit_table(self, input_size) -> tuple[tuple, tuple]:
        logit_x, logit_y = list(), list()
        for i in range(2 ** input_size):
            x = f"{format(i, 'b')}".zfill(input_size)
            x = tuple(int(x) for x in x)
            y = self.logitFunction(x)
            
            logit_x.append(x)
            logit_y.append(y)
        return tuple(logit_x), tuple(logit_y)
    
    @abstractmethod
    def logitFunction(self, input: list) -> int:
        '''
        ex) AND, OR, XOR Gate
        '''
        pass


class AndGate(LogitGate):
    def __init__(self, dataset_size: int, input_size: int=2) -> None:
        super(AndGate, self).__init__(dataset_size, input_size)

    def logitFunction(self, input: list) -> tuple:
        return (int(all(input)), )
    

class OrGate(LogitGate):
    def __init__(self, dataset_size: int, input_size: int=2) -> None:
        super(OrGate, self).__init__(dataset_size, input_size)

    def logitFunction(self, input: list) -> tuple:
        return (int(any(input)), )
    

class XorGate(LogitGate):
    def __init__(self, dataset_size: int, input_size: int=2) -> None:
        super(XorGate, self).__init__(dataset_size, input_size)

    def logitFunction(self, input: list) -> tuple:
        flow_logit, *input = input
        for x in input:
            flow_logit ^= x
        return (flow_logit, )
    
class NotGate(LogitGate):
    def __init__(self, dataset_size: int, input_size: int=2) -> None:
        super(NotGate, self).__init__(dataset_size, input_size)

    def logitFunction(self, input: list) -> tuple:
        return tuple(-x + 1 for x in input)
