from typing import List
import xarray as xr
import torch



class Scaler():
    def __init__(self) -> None:
        pass

    def fit(self, tensor: torch.Tensor):
        raise NotImplementedError

    def transform(self, tensor: torch.Tensor):
        raise NotImplementedError

    def fit_transform(self, tensor: torch.Tensor):
        raise NotImplementedError

    def inverse_transform(self, tensor: torch.Tensor):
        raise NotImplementedError



class StandardScaler(Scaler):
    def __init__(self, 
            src: str,
            drivers: List[str],  
            dtype = torch.float32
        ) -> None:
        super().__init__()

        self._scaler_ds = xr.load_dataset(src)
        self._mean = torch.as_tensor(self._scaler_ds[[f'mean-{drv}' for drv in drivers]].to_array().data, dtype=dtype)
        self._std = torch.as_tensor(self._scaler_ds[[f'std-{drv}' for drv in drivers]].to_array().data, dtype=dtype)

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        scaled_tensor = ((tensor - self._mean) / self._std)
        return scaled_tensor

    def inverse_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        rescaled_tensor = ((tensor * self._std) + self._mean)
        return rescaled_tensor

    def get_mean(self): return self._mean
    def get_std(self): return self._std