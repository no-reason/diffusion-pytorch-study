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