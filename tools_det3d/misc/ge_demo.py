import cv2
import os
from glob import glob
from natsort import natsorted

# image_dir = "./work_dirs/VoD-LGDD_4x4_24e/figures_path/test/det3d"
image_dir = "./work_dirs/VoD-radarpillarnet_4x1_80e/test/det3d"
# output_video = "gt_img.mp4"
# output_video = "LGDD_pred.mp4"
output_video = "baseline_pred.mp4"
fps = 10

filename_suffix = "_det3d_pred.png"
# filename_suffix = "_det3d_gt.png"

all_images = glob(os.path.join(image_dir, f"*{filename_suffix}"))
all_images = natsorted(all_images)

print(f"🖼️ 找到原始图片数量: {len(all_images)}")

# 筛选出第二位编号在 278~04703 范围内的图片
filtered_images = []
for path in all_images:
    filename = os.path.basename(path)
    parts = filename.split("_")
    if len(parts) < 3:
        print(f"❌ 跳过格式错误的文件名: {filename}")
        continue
    try:
        prefix_number = int(parts[1])
        if 278 <= prefix_number <= 4703:
            filtered_images.append(path)
    except ValueError:
        print(f"❌ 非数字前缀：{filename}")
        continue

print(f"✅ 符合条件的图片数量: {len(filtered_images)}")
    
if not filtered_images:
    raise FileNotFoundError("没有找到符合条件的图片：*_det3d_gt.png")

first_image = cv2.imread(filtered_images[0])
height, width, _ = first_image.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

for img_path in filtered_images:
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error：无法读取图片 {img_path}，将跳过。")
        continue
    resized_img = cv2.resize(img, (width, height))
    video_writer.write(resized_img)

video_writer.release()
print(f"✅ 视频保存成功：{output_video}")
