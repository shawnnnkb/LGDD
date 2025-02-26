import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

models = ["PointAugmentin", "RCFusion", "FUTR3D", "BEVFusion",
          "LXL", "PointPillars", "CenterPoint", "PillarNeXt", "RadarPillarNet", 
          "LXL-R", "SMURF", "Ours"]

fps = [6.2, 7.5, 9.1, 5.4, 3.6, 42.0, 34.5, 28.5, 43.9, 36.6, 23.1, 19.7]  # ms
EA_mAP = [28.94, 33.85, 32.42, 32.71, 36.32, 28.31, 29.07, 29.20, 30.37, 30.79, 32.99, 34.02]  # %
DC_mAP = [39.04, 39.76, 37.51, 41.12, 41.20, 36.23, 36.18, 35.71, 39.24, 38.42, 40.98, 42.02]

multi_sensor = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2]
# colors = ["#E57373" if ms == 0 else "#9575CD" if ms == 1 else "#388E3C" for ms in multi_sensor]
colors = ["#E57373" if ms == 0 else "#A388D4" if ms == 1 else "#388E3C" for ms in multi_sensor]

# 组合所有相关数据
combined = list(zip(models, fps, EA_mAP, DC_mAP, multi_sensor, colors))

# 按FPS降序排列（确保大的气泡先绘制）
sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)

# 解包排序后的数据
models_sorted, fps_sorted, EA_mAP_sorted, DC_mAP_sorted, _, colors_sorted = zip(*sorted_combined)

plt.figure(figsize=(10, 6))
sc = plt.scatter(
    EA_mAP_sorted, 
    DC_mAP_sorted,
    s=np.array(fps_sorted) * 50,  # 适当放大尺寸
    c=colors_sorted,
    alpha=1,  # 去除透明度
    edgecolors="k",
    linewidths=2,
    zorder=3
)

# Adding the fps value to each bubble
for i, txt in enumerate(fps_sorted):
    # 智能格式化数值显示
    if txt.is_integer():
        fps_str = f"{int(txt)}"      # 整数去小数点
    else:
        fps_str = f"{txt:.1f}".rstrip('0').rstrip('.')  # 处理类似7.0的特殊情况
    
    if fps_str == '42':
            plt.text(
            EA_mAP_sorted[i]-0.25, 
            DC_mAP_sorted[i],
            fps_str,  # 使用处理后的字符串
            fontsize=6.5,
            ha='center',
            va='center',
            color='white',
            weight='heavy',
            zorder=4
        )
    elif fps_str == '23.1':
            plt.text(
            EA_mAP_sorted[i]+0.25, 
            DC_mAP_sorted[i],
            fps_str,  # 使用处理后的字符串
            fontsize=6.5,
            ha='center',
            va='center',
            color='white',
            weight='heavy',
            zorder=4
        )
    elif fps_str == '34.5':
            plt.text(
            EA_mAP_sorted[i], 
            DC_mAP_sorted[i]+0.1,
            fps_str,  # 使用处理后的字符串
            fontsize=6.5,
            ha='center',
            va='center',
            color='white',
            weight='heavy',
            zorder=4
        )
    else:
        plt.text(
            EA_mAP_sorted[i], 
            DC_mAP_sorted[i],
            fps_str,  # 使用处理后的字符串
            fontsize=6.5,
            ha='center',
            va='center',
            color='white',
            weight='heavy',
            zorder=4
        )


plt.rcParams['mathtext.fontset'] = 'stix'  # 使用STIX字体
plt.rcParams['font.family'] = 'STIXGeneral'  # 配套字体设置
# Labels and title
plt.xlabel(r"$\mathrm{mAP}_{\mathrm{3D}}$ [%]")
plt.ylabel(r"$\mathrm{mAP}_{\mathrm{BEV}}$ [%]")

x_ticks = np.arange(26, 38, 2)
y_ticks = np.arange(35, 43, 1)
plt.xticks(x_ticks)
plt.yticks(y_ticks)

plt.gca().set_aspect(2, adjustable='box')  # This makes the horizontal distance smaller
plt.grid(True, linestyle="--", alpha=0.6, zorder=0)

# Legend
import matplotlib.patches as mpatches
single_patch = mpatches.Patch(color="#E57373", label="radar + camera")
multi_patch = mpatches.Patch(color="#A388D4", label="radar")
ours_patch = mpatches.Patch(color="#388E3C", label="radar (ours)")
plt.legend(handles=[single_patch, multi_patch, ours_patch], loc="lower right")

plt.savefig("./docs/plot_tools/TJ4D-bubble_chart.png", dpi=300, bbox_inches="tight")
