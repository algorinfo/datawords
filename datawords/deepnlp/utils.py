from typing import List, Optional

from pydantic import BaseModel, Field


class GPUInfo(BaseModel):
    device_count: int = 0
    devices: List[str] = Field(default_factory=list)


def get_gpu_info(framework="pt") -> Optional[GPUInfo]:
    import torch

    available = torch.cuda.is_available()
    if not available:
        return None
    count = torch.cuda.device_count()
    devices = [torch.cuda.get_device_name(i) for i in range(0, count)]
    return GPUInfo(device_count=count, devices=devices)
