# python 学习

## argparse 模块
argparse这个库可以实现命令行参数的解析和处理

这样在命令行运行python文件的时候就可以直接传递参数了

下面是常见的用法:
```python
import argparse # 导入argparse这个库

def parse_args():          # ← 定义函数，不会自动执行
    p = argparse.ArgumentParser()
    p.add_argument(...)
    return p.parse_args()

def main():
    args = parse_args()    # ← main 里手动调用一次
    ...

if __name__ == "__main__":
    main()                 # ← 终端运行时 → main() → parse_args()


```


## Python中类的三种方法

- 普通def定义
- @classmethod定义
- @staticmethod定义

```python
# -*- coding: utf-8 -*-
# 示例：用 Diffusion 模型里的「Beta 调度器」（你训练里肯定要用到的东西）演示三种方法
class BetaScheduler:
    """扩散模型的 Beta 调度器：负责生成 timestep 对应的噪声强度 beta"""
    
    # === 类属性（所有实例共享的常量）===
    DEFAULT_TIMESTEPS = 1000  # 默认总步数，类级别共享
    
    # ==================== 1. 普通 def → 实例方法 ====================
    # 【必须传 self】：因为要访问「当前这个具体调度器实例」的专属状态
    def __init__(self, start_beta: float = 0.0001, end_beta: float = 0.02, timesteps: int = None):
        # self 是「当前实例」，self.xxx 是实例专属属性（和其他实例不共享）
        self.start = start_beta
        self.end = end_beta
        self.timesteps = timesteps or self.DEFAULT_TIMESTEPS  # 可以通过 self 访问类属性
        # 提前预计算所有 timestep 的 beta（这是实例的核心状态）
        self.betas = [self.start + (self.end - self.start) * i / (self.timesteps - 1) for i in range(self.timesteps)]
    
    # 【实例方法的典型应用】：依赖实例专属的 self.betas 状态
    def get_beta_t(self, t: int) -> float:
        """获取第 t 步的 beta 值（扩散训练里每次前向都要调）"""
        return self.betas[t]

    # ==================== 2. @classmethod → 类方法 ====================
    # 【必须传 cls】：cls 是「类本身」（不是实例！），用来做「备用构造函数」/「类级别的工具」
    # 核心价值：支持继承——子类调用时，cls 自动变成子类，不会写死父类
    @classmethod
    def from_config(cls, config: dict) -> "BetaScheduler":
        """从配置字典快速创建调度器实例（对应 PyTorch 里的 from_pretrained）"""
        print(f"[类方法] 当前 cls = {cls.__name__}  ← 这里 cls 是类本身，不是实例！")
        # 用 cls() 创建实例，而不是写死 BetaScheduler() → 子类继承时自动适配
        return cls(
            start_beta=config.get("start", 0.0001),
            end_beta=config.get("end", 0.02),
            timesteps=config.get("timesteps", cls.DEFAULT_TIMESTEPS)
        )

    # ==================== 3. @staticmethod → 静态方法 ====================
    # 【不用 self/cls】：纯数学工具，逻辑属于「Beta 调度器」但不依赖任何实例/类状态
    # （对应你脚本里的 rotation_matrix，或者 PyTorch 里的 torch.cat 这种工具函数）
    @staticmethod
    def linear_interpolate(a: float, b: float, ratio: float) -> float:
        """纯数学工具：线性插值（扩散里调整噪声强度的常用操作）"""
        ratio = max(0.0, min(1.0, ratio))  # 截断到 [0,1]
        return a + (b - a) * ratio

# ==================== 子类演示（专门突出 @classmethod 的继承优势）====================
class CosineBetaScheduler(BetaScheduler):
    """余弦 Beta 调度器（比线性调度更平滑）—— 继承自 BetaScheduler"""
    # 重写静态方法：余弦调度的插值逻辑和线性不同
    @staticmethod
    def linear_interpolate(a: float, b: float, ratio: float) -> float:
        """余弦插值（纯数学工具，和类状态无关，所以用静态方法）"""
        ratio = max(0.0, min(1.0, ratio))
        cos_ratio = 0.5 * (1 - __import__("math").cos(ratio * __import__("math").pi))
        return a + (b - a) * cos_ratio

# ==================== 演示区（直接运行就能看结果）====================
if __name__ == "__main__":
    print("="*50)
    print("1. 【普通实例方法】调用：必须通过实例，依赖 self 状态")
    base_scheduler = BetaScheduler()  # 先创建实例（绑定 self）
    print(f"  第 500 步的 beta：{base_scheduler.get_beta_t(500):.6f}\n")

    print("="*50)
    print("2. 【@classmethod 类方法】调用：可以直接通过类/子类，支持继承适配")
    # 用类方法创建基类实例
    base_from_config = BetaScheduler.from_config({"start": 0.0001, "end": 0.02})
    print(f"  基类 from_config 创建的实例类型：{type(base_from_config).__name__}")
    # 重点：子类调用类方法时，cls 自动变成子类！（这就是 @classmethod 的核心价值）
    cos_scheduler = CosineBetaScheduler.from_config({"start": 0.0001, "end": 0.02})
    print(f"  子类 from_config 创建的实例类型：{type(cos_scheduler).__name__}（如果写死 BetaScheduler() 就会错！）\n")

    print("="*50)
    print("3. 【@staticmethod 静态方法】调用：不用实例/类，纯工具")
    # 可以通过类名直接调，也可以通过实例调（推荐类名，语义更清晰）
    linear_val = BetaScheduler.linear_interpolate(0.0001, 0.02, 0.5)
    cos_val = CosineBetaScheduler.linear_interpolate(0.0001, 0.02, 0.5)
    print(f"  线性插值 50% 噪声：{linear_val:.6f}")
    print(f"  余弦插值 50% 噪声：{cos_val:.6f}（子类重写了静态方法，逻辑不同）")
```





## assert 语句
相当于判断异常用的一个语句
```python
assert extra_feature_channels >= 0
# 如果 extra_feature_channels 小于 0，就会抛出AssertionError异常
# 否则程序就会继续执行
```