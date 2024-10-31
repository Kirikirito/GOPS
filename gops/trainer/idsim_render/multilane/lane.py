import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from shapely import offset_curve
from matplotlib.patches import Polygon
from scipy.interpolate import interp1d
from idscene.network import SumoNetwork
from gops.trainer.idsim_render.render_params import theme_color_dict


def equalize_line_lengths(xy1, xy2):
    # 计算两条线的长度
    xy1 = np.array(xy1)
    xy2 = np.array(xy2)

    
    # 先转置以匹配 interp1d 需求，使点的顺序在第一个维度
    xy1 = xy1.T
    xy2 = xy2.T

    # 计算两条线的长度
    n_points1 = xy1.shape[0]
    n_points2 = xy2.shape[0]

    # 新的公共长度
    common_length = max(n_points1, n_points2)

    # 创建新的索引
    new_indices = np.linspace(0, 1, common_length)

    # 插值函数，axis=0 表明沿着点的顺序维度进行插值
    interp1 = interp1d(np.linspace(0, 1, n_points1), xy1, axis=0, kind='linear')
    interp2 = interp1d(np.linspace(0, 1, n_points2), xy2, axis=0, kind='linear')

    # 插值得到新的线段
    new_xy1 = interp1(new_indices)
    new_xy2 = interp2(new_indices)

    return new_xy1, new_xy2

def plot_lane_lines(ax: Axes, network: SumoNetwork, zorder: float,factor=1):
    theme_color = theme_color_dict['light']
    lane_width = 3.75*factor
    lane_center_lines = list(network._center_lines.values())
    lane_center_lines.insert(0, offset_curve(
        lane_center_lines[0], -lane_width))
    for i, line in enumerate(lane_center_lines):
        if i <= len(lane_center_lines) // 2:
            offset = lane_width / 2
        else:
            offset = -lane_width / 2
        lane_center_lines[i] = offset_curve(line, offset)



    # Add lane areas
    xy1 = lane_center_lines[0].xy
    xy2 = lane_center_lines[4].xy
    xy1, xy2 = equalize_line_lengths(xy1, xy2)
    vertices = np.concatenate([xy1*factor, xy2*factor], axis=0)
    # Set ax background for areas outside the lanes
    ax.set_facecolor(theme_color['background'])
    



    p = Polygon(vertices, closed=True, color='grey', zorder=zorder-1)
    ax.add_patch(p)
 # Add lane lines and boundaries
    for i, v in enumerate(lane_center_lines):
        x, y = v.xy
        x= x*factor
        y= y*factor
        if i == len(lane_center_lines) // 2:
            # Center line (double yellow solid line)
            left_yellow_line = offset_curve(v, lane_width / 10)
            right_yellow_line = offset_curve(v, -lane_width / 10)
            x_left, y_left = left_yellow_line.xy
            x_right, y_right = right_yellow_line.xy
            ax.add_line(Line2D(x_left, y_left, linewidth=2, color="yellow", zorder=zorder))
            ax.add_line(Line2D(x_right, y_right, linewidth=2, color="yellow", zorder=zorder))
            continue
        if i in [0, len(lane_center_lines) // 2, len(lane_center_lines) // 2 + 1]:
            # Boundary and division lines (double white solid lines for opposite traffic division)
            ax.add_line(Line2D(x, y, linewidth=2, color="white", zorder=zorder))
        else:
            # Regular lane lines (dashed white lines)
            ax.add_line(Line2D(x, y, linewidth=2, color="white", linestyle=(0, (10, 10)), zorder=zorder))
