# 说明

## Table123
替换对应 yaml 文件训练即可

## Table4
第一行就是 Baseline  
第四行就是 Best

第二行 FLIR_Baseline+CSAF  
模块改成下面的：

```python
class FDBlock(nn.Module):        
    def __init__(self, dim: int = 0):
        super().__init__()

    def forward(self, input: torch.Tensor):
        a0 = input[0]
        b0 = input[1]

        return a0 + b0
```

第三行：FLIR_Baseline+CSAF+CFMx2+FAEFx0  
模块如下：

```python
class CMF1(nn.Module):        
    def __init__(self, dim: int = 0):
        super().__init__()
        self.fusion = nn.Sequential(*[SpatialCross(dim) for _ in range(4)])

    def forward(self, input: torch.Tensor):
        a0 = input[0]
        b0 = input[1]
        a, b = self.fusion([a0, b0])
        return a, b


class CMF2(nn.Module):
    def __init__(self, dim=0):
        super().__init__()
        self.fusion1 = nn.Sequential(*[Fusion1(dim, dim) for _ in range(2)])
        # self.fusion2 = nn.Sequential(*[Fusion2(dim) for _ in range(2)])

    def forward(self, input):
        a0, b0 = input
        a, b = self.fusion1([a0, b0])
        # a, b = self.fusion2([a, b])
        return a + b
```

## Table5
同 Table4

## Table6

**第 1 行 CMF*1 模块设置**

```python
class CMF2(nn.Module):
    def __init__(self, dim=0):
        super().__init__()
        self.fusion1 = nn.Sequential(*[Fusion1(dim, dim) for _ in range(1)])
        self.fusion2 = nn.Sequential(*[Fusion2(dim) for _ in range(2)])

    def forward(self, input):
        a0, b0 = input
        a, b = self.fusion1([a0, b0])
        a, b = self.fusion2([a, b])
        return a + b
```

**第 2 行就是 Best**

**第 3 行 CMF*3 模块设置**

```python
class CMF2(nn.Module):
    def __init__(self, dim=0):
        super().__init__()
        self.fusion1 = nn.Sequential(*[Fusion1(dim, dim) for _ in range(1)])
        self.fusion2 = nn.Sequential(*[Fusion2(dim) for _ in range(2)])

    def forward(self, input):
        a0, b0 = input
        a, b = self.fusion1([a0, b0])
        a, b = self.fusion2([a, b])
        return a + b
```

## Table7
同上修改，只不过修改第二个（fusion2）

## Table8
在 `FAE-Det/ultralytics/nn/modules/block.py` 中替换原来的模块

- 第 2 行 使用模块 `class Fusion2(nn.Module):`  # = Table8-2
- 第 3 行 使用模块 `class Fusion2(nn.Module):`  # = Table8-3
- 第 4 行 使用模块 `class Fusion2(nn.Module):`  # = Table8-4

## Table9
在 `FAE-Det/ultralytics/nn/modules/block.py` 中替换原来的模块

- 第 2 行 使用模块 `class Fusion2(nn.Module):`  # = Table9-2
- 第 3 行 使用模块 `class Fusion2(nn.Module):`  # = Table9-3

## Table10
需要同时替换模块和 yaml 文件

## Table11
训练时需要换对应方法的 yaml 文件，模块已写入

## Table12
模块无需修改，训练时需要调整模型大小