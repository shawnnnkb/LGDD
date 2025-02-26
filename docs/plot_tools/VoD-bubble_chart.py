import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

models = ["FUTR3D", "BEVFusion", "RCFusion", "PointAugmenting", 
          "LXL", "PillarNeXt", "PointPillars", "CenterPoint", "RadarPillarNet", 
          "LXL-R", "SMURF", "ours"]

fps = [11.0, 7.1, 9.0, 7.9, 6.1, 31.6, 101.4, 38.3, 54.6, 44.7, 30.3, 25.1]  # ms
EA_mAP = [49.03, 49.25, 49.65, 52.60, 56.31, 42.23, 45.18, 45.42, 46.01, 46.84, 50.97, 53.55]  # %
DC_mAP = [69.32, 68.52, 69.23, 69.06, 72.93, 63.61, 67.48, 65.06, 65.86, 68.51, 69.72, 72.90]

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
    s=np.array(fps_sorted) * 35,  # 适当放大尺寸
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
    
    if fps_str == '11':
            plt.text(
            EA_mAP_sorted[i]-0.15, 
            DC_mAP_sorted[i]+0.05,
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
plt.xlabel(r"$\mathrm{mAP}_{\mathrm{EAA}}$ [%]")
plt.ylabel(r"$\mathrm{mAP}_{\mathrm{DC}}$ [%]")

x_ticks = np.arange(41, 58, 2)
y_ticks = np.arange(63, 74, 1)
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

plt.savefig("./docs/plot_tools/VoD-bubble_chart.png", dpi=300, bbox_inches="tight")
