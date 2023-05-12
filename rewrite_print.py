import os

# 保存原始的print函数，以便稍后调用它。
rewrite_print = print

# 定义新的print函数。
def print(*arg):
   # 首先，调用原始的print函数将内容打印到控制台。
   rewrite_print(*arg)
   
   # 如果日志文件所在的目录不存在，则创建一个目录。
   output_dir = "log_file"
   if not os.path.exists(output_dir):
         os.makedirs(output_dir)
   
   # 打开（或创建）日志文件并将内容写入其中。
   log_name = 'log.txt'
   filename = os.path.join(output_dir, log_name)
   rewrite_print(*arg,file=open(filename,"a"))

def print_acc(*arg):
   # 首先，调用原始的print函数将内容打印到控制台。
   rewrite_print(*arg)
   
   # 如果日志文件所在的目录不存在，则创建一个目录。
   output_dir = "log_file"
   if not os.path.exists(output_dir):
         os.makedirs(output_dir)
   
   # 打开（或创建）日志文件并将内容写入其中。
   log_name = 'log_acc.txt'
   filename = os.path.join(output_dir, log_name)
   rewrite_print(*arg,file=open(filename,"a"))