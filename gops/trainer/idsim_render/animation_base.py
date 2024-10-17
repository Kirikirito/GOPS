import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

from gops.trainer.idsim_idc_evaluator import EvalResult
from gops.trainer.idsim_render.color import SUR_COLOR, SUR_COLOR_WITH_ALPHA, SUR_FOCUS_COLOR, SUR_FOCUS_COLOR_WITH_ALPHA, SUR_COLOR_TRANSPARENT
from gops.trainer.idsim_render.process_fcd import FCDLog
from gops.trainer.idsim_render.render_params import multilane_surr_size_dict, crossroad_surr_size_dict, traffic_light_length, traffic_light_width, \
    sur_face_color, ego_face_color, ref_color_list
from idsim.utils.coordinates_shift import convert_sumo_coord_to_ground_coord
import matplotlib.pyplot as plt
import numpy as np
from idscene.network import SumoNetwork
from idscene.scenario import ScenarioData
from idsim_model.params import model_config
from matplotlib.animation import FFMpegWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon, Rectangle as Rt
from shapely import offset_curve


def veh2vis(xy: np.ndarray, phi: float, veh_length: float, veh_width: float):
    R = np.array([[np.cos(phi), np.sin(phi)],
                  [-np.sin(phi), np.cos(phi)]])
    xy_vis = xy - np.array([veh_length / 2, veh_width / 2]) @ R
    return tuple(xy_vis)


def rad_to_deg(rad: float) -> float:
    return rad * 180 / np.pi


def create_veh(
    ax: plt.Axes,
    xy: Tuple[float, float],
    phi: float,
    length: float,
    width: float,
    facecolor: str = None,
    edgecolor: str = None,
    zorder=1,
) -> Rt:
    vehPatch = ax.add_patch(Rt(
        (0, 0), length, width, angle=0., facecolor=facecolor, edgecolor=edgecolor, zorder=zorder, linewidth=1
    ))
    update_veh(vehPatch, xy, phi, length, width)
    return vehPatch


def update_veh(vehPatch: Rt, xy: Tuple[float, float], phi: float, length: float, width: float):
    xy_vis = veh2vis(xy, phi, length, width)
    vehPatch.set_xy(xy_vis)
    vehPatch.set_angle(rad_to_deg(phi))


def remove_veh(vehPatch: Rt):
    vehPatch.remove()


class AnimationBase:
    def __init__(self, theme_style='light', fcd_file=None, config=None, env_scenario=None) -> None:
        self.theme_style = theme_style
        self.fcd_log = FCDLog.from_file(fcd_file)
        self.config = config

        self.last_veh_id_list = []
        self.last_cyl_id_list = []
        self.last_ped_id_list = []
        self.surr_dict = {}
        self.surr_head_dict = {}
        self.surr_focus_list = []
        self.surr_focus_ref_list = []

        self.REF_POINT_NUM = 31
        self.REF_LINEWIDTH = 2.0

        self.env_scenario = env_scenario if env_scenario is not None else "multilane"
        assert self.env_scenario in ["multilane", "crossroad"], ("invalid scenario type " + str(self.env_scenario))

    def plot_traffic(self, episode_data, fig, gs):
        pass

    def plot_figures(self, episode_data, fig, gs):
        pass

    def generate_animation(self,
                           episode_data: EvalResult,
                           save_path: Path,
                           episode_index: int,
                           fps=10,
                           mode='debug'
                           ):
        pass

    def clear_all_list(self):
        self.last_veh_id_list = []
        self.last_cyl_id_list = []
        self.last_ped_id_list = []
        self.surr_dict = {}
        self.surr_head_dict = {}
        self.surr_focus_list = []
        self.surr_focus_ref_list = []

    def update_sur_participants(self, ax, cur_time, episode_data, step):
        cur_time = max(0, cur_time - 0.1)  # 
        participants = self.fcd_log.at(cur_time).vehicles
        ego_veh_id = episode_data.ego_id
        if self.env_scenario == "multilane": #TODO: update when change the scenario setting
            veh_id_list = [p.id for p in participants if p.id !=
                ego_veh_id and (p.type.startswith('v'))]
            cyl_id_list = [p.id for p in participants if p.id !=
                        ego_veh_id and p.type == 'b1']
            ped_id_list = [p.id for p in participants if p.id !=
                        ego_veh_id and p.type == 'person']
        elif self.env_scenario == "crossroad":
            veh_id_list = [p.id for p in participants if p.id !=
                        ego_veh_id and p.type.startswith('v')]
            cyl_id_list = [p.id for p in participants if p.id !=
                        ego_veh_id and p.type == 'b1']
            ped_id_list = [p.id for p in participants if p.id !=
                        ego_veh_id and p.type == 'person']
        else:
            raise ValueError("invalid scenario type " + str(self.env_scenario))
        # veh
        to_add_veh = list(set(veh_id_list) - set(self.last_veh_id_list))
        to_update_veh = list(set(veh_id_list) - set(to_add_veh))
        to_remove_veh = list(set(self.last_veh_id_list) - set(veh_id_list))
        # cyl
        to_add_cyl = list(set(cyl_id_list) - set(self.last_cyl_id_list))
        to_update_cyl = list(set(cyl_id_list) - set(to_add_cyl))
        to_remove_cyl = list(set(self.last_cyl_id_list) - set(cyl_id_list))
        # ped
        to_add_ped = list(set(ped_id_list) - set(self.last_ped_id_list))
        to_update_ped = list(set(ped_id_list) - set(to_add_ped))
        to_remove_ped = list(set(self.last_ped_id_list) - set(ped_id_list))

        to_add = to_add_veh + to_add_cyl + to_add_ped
        to_update = to_update_veh + to_update_cyl + to_update_ped
        to_remove = to_remove_veh + to_remove_cyl + to_remove_ped

        for surr in participants:
            if self.env_scenario == "multilane":  #TODO: update when change the scenario setting
                veh_type = surr.type.split('-')[0]
                assert veh_type in ['v1','v2','v3', 'person','vm','vs', 'vss','b1', 'person'], ('invalid traffic type ' + str(surr.type))

                length, width = multilane_surr_size_dict[veh_type]
            elif self.env_scenario == "crossroad":
                assert surr.type in ['v1', 'v2', 'v3', 'b1', 'person'], ('invalid traffic type ' + str(surr.type))
                length, width = crossroad_surr_size_dict[surr.type]
            else:
                raise ValueError("invalid scenario type " + str(self.env_scenario))
            x, y, phi = convert_sumo_coord_to_ground_coord(
                surr.x, surr.y, surr.angle, length)
            if surr.id in to_add:
                if hasattr(episode_data, 'focused_vehicle_ids') and surr.id in episode_data.focused_vehicle_ids[step]:
                    facecolor = SUR_COLOR_TRANSPARENT
                    edgecolor = SUR_COLOR_TRANSPARENT
                else:
                    facecolor = SUR_COLOR_WITH_ALPHA
                    edgecolor = SUR_COLOR
                self.surr_dict[surr.id] = create_veh(
                    ax, (x, y), phi, length, width,
                    facecolor=facecolor, edgecolor=edgecolor, zorder=200
                )
                self.surr_head_dict[surr.id] = ax.plot(
                    [x, x+length*0.8*np.cos(phi)], [y, y+length*0.8*np.sin(phi)],
                    color=facecolor, linewidth=0.5, zorder=200
                )
            elif surr.id in to_update:
                update_veh(self.surr_dict[surr.id], (x, y), phi, length, width, )
                self.surr_head_dict[surr.id][0].set_data(
                    [x, x + length * 0.8 * np.cos(phi)], [y, y + length * 0.8 * np.sin(phi)])
            elif surr.id == ego_veh_id:
                pass
            else:
                raise ValueError('invalid id')
        for surr in to_remove:
            remove_veh(self.surr_dict[surr])
            del self.surr_dict[surr]
        self.last_veh_id_list = veh_id_list
        self.last_cyl_id_list = cyl_id_list
        self.last_ped_id_list = ped_id_list

        # ---------------- update detect sur participants------------------
        # p [2N+1, num_veh, feature_dim]
        # feature [x, y, phi, speed, length, width, mask]
        surr_states = episode_data.surr_state_list[step][0]
        surr_param = episode_data.surr_state_list[step][:self.REF_POINT_NUM]
        for surr in self.surr_focus_list:
            surr.remove()
        for surr_ref in self.surr_focus_ref_list:
            surr_ref.remove()
        self.surr_focus_list = []
        self.surr_focus_ref_list = []
        if len(surr_states.shape) == 1:  # NOTE: when num_veh=1, surr_states.shape=(feature_dim,)
            surr_states = surr_states.reshape(1, -1)
        for i in range(surr_states.shape[0]):
            x, y, phi, speed, length, width, mask = surr_states[i]
            if episode_data.attn_weight_list != []:
                attn_weight = episode_data.attn_weight_list[step][i]
            else:
                attn_weight = 1.0
            if mask == 1:
                # print(f"attn_weight: {attn_weight}")
                facecolor_with_attn = self.get_color_by_attn_weight(attn_weight)
                edgecolor_with_attn = SUR_FOCUS_COLOR
                self.surr_focus_list.append(create_veh(ax, (x,y), phi, length, width, facecolor=facecolor_with_attn, edgecolor=edgecolor_with_attn, zorder=201))
                self.surr_focus_ref_list.append(ax.add_line(Line2D(
                        surr_param[:, i, 0], surr_param[:, i, 1],
                        color=SUR_FOCUS_COLOR_WITH_ALPHA, linewidth=self.REF_LINEWIDTH, zorder=201
                    )))
                
    def get_color_by_attn_weight(self, attn_weight):
        # green if attn_weight<0.15
        if attn_weight < 0.15:
            return (0, 1, 0, 0.5+ 0.5*attn_weight/0.15)
        # yellow if 0.15<=attn_weight<0.5
        elif attn_weight < 0.5:
            return (1, 1, 0, 0.5+ 0.5*(attn_weight-0.15)/0.35)
        # red if 0.5<=attn_weight
        else:
            return (1, 0, 0, 0.5+ 0.5*(attn_weight-0.5)/0.5)
