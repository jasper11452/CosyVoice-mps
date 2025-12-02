import os

# 路径可能需要根据你的实际安装位置调整
file_path = '/Users/jasper/CosyVoice/cosyvoice/flow/decoder.py'

if not os.path.exists(file_path):
    print(f"错误: 找不到文件 {file_path}")
    exit(1)

with open(file_path, 'r') as f:
    content = f.read()

if 'class SafeMish' in content:
    print("文件似乎已经修补过了。")
    exit(0)

print("正在修补 cosyvoice/flow/decoder.py ...")

# 1. 定义 SafeMish 类代码
safe_mish_code = """
class SafeMish(nn.Module):
    def forward(self, x):
        if x.device.type == 'mps':
            return x * torch.tanh(torch.logaddexp(x, torch.zeros_like(x)))
        return torch.nn.functional.mish(x)
"""

# 2. 将类定义插入到 import 之后 (简单的插入到文件开头 import 块之后)
# 我们假设文件开头有一些 import，我们在找到第一个 'class ' 定义前插入
insert_pos = content.find('class ')
if insert_pos == -1:
    print("无法定位插入点")
    exit(1)

new_content = content[:insert_pos] + safe_mish_code + "\n\n" + content[insert_pos:]

# 3. 替换 nn.Mish()
if 'nn.Mish()' not in new_content:
    print("警告: 未在文件中找到 nn.Mish()，可能使用了不同的写法。")
else:
    new_content = new_content.replace('nn.Mish()', 'SafeMish()')

with open(file_path, 'w') as f:
    f.write(new_content)

print("修补完成！请重新运行推理测试。")