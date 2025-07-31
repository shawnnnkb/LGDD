import os
import time
from datetime import datetime, timedelta

# 指定目标文件夹
target_dir = "/ssd/home/bxk/CODE-40902-PhD-2/LGDD/work_dirs/VoD-LGDD_4x4_24e/figures_path/test/det3d"

# 获取今天 0 点的时间戳
today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()

# 遍历文件夹中的所有文件
for filename in os.listdir(target_dir):
    file_path = os.path.join(target_dir, filename)

    if os.path.isfile(file_path):
        # 获取文件创建时间（Linux 上实际是 ctime，更多是修改时间）
        file_ctime = os.path.getctime(file_path)

        # 如果文件创建时间早于今天
        if file_ctime < today_start:
            print(f"删除：{file_path}")
            os.remove(file_path)
