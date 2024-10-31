import os
import pickle
from pathlib import Path

from gops.trainer.idsim_idc_mf_multilane_evaluator import EvalResult
from gops.trainer.idsim_render.color import EGO_COLOR, SUR_COLOR, EGO_COLOR_WITH_ALPHA, SUR_COLOR_WITH_ALPHA, \
    TRAJ_COLOR_WITH_ALPHA
from gops.trainer.idsim_render.multilane.info_bar import plot_scale, plot_speed_dashboard, plot_action_bar, \
    plot_value_bar
from gops.trainer.idsim_render.multilane.lane import plot_lane_lines
from gops.trainer.idsim_render.plot_mf_time_flow import plot_action, plot_reward, plot_y_ref_phi_ref, plot_vx_vy_r
from matplotlib import gridspec
from matplotlib.figure import Figure

from gops.trainer.idsim_render.animation_base import AnimationBase, create_veh, remove_veh, update_veh

import matplotlib.pyplot as plt
import numpy as np
from idscene.scenario import ScenarioData
from matplotlib.animation import FFMpegWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle as Rt

VEH_LENGTH = 5.0
VEH_WIDTH = 1.8

# REF_POINT_NUM = 31
# REF_LINEWIDTH = 2.0
# REF_ALPHA = 0.3


class AnimationLane(AnimationBase):
    def __init__(self, theme_style, fcd_file, config) -> None:
        super().__init__(theme_style, fcd_file, config, "multilane")
        self.ax1_1, self.ax1_2, self.ax1_3 = None, None, None
        self.ax2_1, self.ax2_2, self.ax2_3, self.ax2_4 = None, None, None, None
        self.ax3_1 = None
        self.ax4_1, self.ax4_2 = None, None
        self.ref_artist_list = []
        self.scale_artist_list = []
        self.background_artist_list = []
        self.speed_dashboard_artist_list = []
        self.action_bar_artist_list = []
        self.value_bar_artist_list = []
        self.task_name = "multilane"
        downsample_ref_point_index = self.config['env_model_config'].get('downsample_ref_point_index', None)
        if downsample_ref_point_index is not None:
            self.ref_points_num = len(downsample_ref_point_index)
        else:
            self.ref_points_num = self.config['env_model_config']['N'] +1 

    def clear_all_list(self):
        super().clear_all_list()
        self.ref_artist_list = []
        self.scale_artist_list = []
        self.background_artist_list = []
        self.speed_dashboard_artist_list = []
        self.action_bar_artist_list = []
        self.value_bar_artist_list = []


    def generate_animation(
        self,
        episode_data: EvalResult,
        save_path: Path,
        episode_index: int,
        test_scene: str = None,
        frame_skip=1,
        fps=20,
        mode='debug',
        dpi = 600,
        max_frame_num = 2000,
        plot_reward=True,
    ):
        metadata = dict(title='Demo', artist='Guojian Zhan', comment='idsim')
        writer = FFMpegWriter(fps=fps, metadata=metadata)
        save_video_path = save_path / 'video'
        os.makedirs(save_video_path, exist_ok=True)

        # ------------------ initialization -------------------
        # clear all list
        self.clear_all_list()


        # remove empty list in episode_data.reward_info
        empty_keys = []
        for k, v in episode_data.reward_info.items():
            if len(v) == 0:
                empty_keys.append(k)
        for k in empty_keys:
            del episode_data.reward_info[k]

        if len(episode_data.time_stamp_list) <=1:
            print('no data to plot')
            return
        # create figure
        fig: Figure = plt.figure(figsize=(20, 10), dpi=dpi)
        gs = gridspec.GridSpec(4, 2)

        print(f'Initializing traffic...')
        ax = self.plot_traffic(episode_data, fig, gs)

        print(f'Plotting figures...')
        eval_dict = self.plot_figures(episode_data, fig, gs)

        # create ego vehicle
        ego_veh = create_veh(ax, (0, 0), 0, VEH_LENGTH,
                             VEH_WIDTH, facecolor=EGO_COLOR_WITH_ALPHA, edgecolor=EGO_COLOR)
        if test_scene is not None:
            ax.text( # test_scene
                0.01, 0.95, f'test_scene: {test_scene}', fontsize=10,
                horizontalalignment='left', verticalalignment='top',
                transform=ax.transAxes
            )
            video_name = f'{self.task_name}_{episode_index}_{test_scene}_frame_skip_{frame_skip}.mp4'
        else:
            video_name = f'{self.task_name}_{episode_index}_frame_skip_{frame_skip}.mp4'
        
        writer.setup(fig, os.path.join(save_video_path, video_name), dpi=dpi)

        if episode_data.paths_value_list:    
            value = episode_data.paths_value_list
        else:
            value = episode_data.value_list
        value = np.array(value) if value else None
        min_value = np.min(value) if value is not None else 0.0
        max_value = np.max(value) if value is not None else 0.0

        # plot reward text info
        reward_text_colors = [line.get_color() for line in self.ax3_1.get_lines()]
        reward_text_labels = self.ax3_1.get_legend_handles_labels()[1]
        reward_text_values = [line.get_ydata() for line in self.ax3_1.get_lines()]
        text_handles = []
        # ---------------------- update-----------------------
        max_frame_num = min(max_frame_num, len(episode_data.time_stamp_list))
        for step in range(0, max_frame_num, frame_skip):
            if step == frame_skip or step % (max_frame_num // 10) == 0:
                #save the first frame as svg and jpg
                fig.savefig(save_path / 'video' / f'{video_name}.svg', format='svg')
                fig.savefig(save_path / 'video' / f'{video_name}.jpg', format='jpg')
            if mode == 'debug' and step % 10 == 0:
                print(f'step={step}/{len(episode_data.time_stamp_list)}')
            cur_time = episode_data.time_stamp_list[step]
            cur_time = round(cur_time * 10) / 10

            # # ---------------- set limit of figure ------------------
            self.adjust_lim(episode_data, eval_dict, step)

            # # ---------------- update ego veh------------------
            ego_x, ego_y, _, _, ego_phi, _ = episode_data.ego_state_list[step]
            circle = Circle((ego_x, ego_y), 1.0, facecolor=TRAJ_COLOR_WITH_ALPHA, edgecolor='none')
            ax.add_patch(circle)
            update_veh(ego_veh, (ego_x, ego_y), ego_phi, VEH_LENGTH, VEH_WIDTH)

            # # ---------------- update ego ref------------------
            self.update_ego_ref(ax, episode_data, step)

            # # ---------------- update sur participants------------------
            self.update_sur_participants(ax, cur_time, episode_data, step)

            # set center
            screen_width, screen_height = 100, 125
            center_x, center_y = ego_x, ego_y - \
                (screen_height - screen_width) / 2
            ax.set_xlim(center_x - screen_width / 2,
                        center_x + screen_width / 2)
            ax.set_ylim(center_y - screen_height / 2,
                        center_y + screen_height / 2)

            # plot scale
            for scale in self.scale_artist_list:
                scale.remove()
            self.scale_artist_list = plot_scale(
                ax=ax,
                xy=(center_x - screen_width / 2 + 3,
                    ego_y - screen_width / 2 + 2),
                zorder=3,
            )

            # plot background of speed dashboard and action bar
            margin = 5
            for background in self.background_artist_list:
                background.remove()
            self.background_artist_list = [ax.add_patch(Rt(
                (center_x - screen_width / 2 - margin,
                 center_y - screen_height / 2 - margin),
                screen_width + 2 *
                margin, (screen_height - screen_width) + margin,
                facecolor="white", edgecolor="black", zorder=2
            ))]

            # plot speed dashboard
            speed = episode_data.ego_state_list[step][2] * 3.6
            for artist in self.speed_dashboard_artist_list:
                artist.remove()
            self.speed_dashboard_artist_list = plot_speed_dashboard(
                ax=ax,
                xy=(center_x - 33, center_y - screen_height / 2 + 10),
                speed=speed,
                zorder=3,
            )
            
            # plot reward text info
            if plot_reward:
                reward_text = ''
                if step > 0:
                    for handle in text_handles:
                        handle.remove()
                    text_handles = []
                for i in range(len(reward_text_labels)):
                    reward_text = f'{reward_text_labels[i]}: {reward_text_values[i][step]:.2f}\n'
                    x_pos = center_x + screen_width / 2 - 30
                    y_pos = center_y - i * 2
                    text_handles.append(ax.text(x_pos,y_pos, reward_text, color=reward_text_colors[i], fontsize=10))



            # plot action bar
            action = episode_data.action_real_list[step]
            action[1] = -action[1]  # steer, turn right is positive
            for artist in self.action_bar_artist_list:
                artist.remove()
            self.action_bar_artist_list = plot_action_bar(
                ax=ax,
                xy=(center_x - 12, center_y - screen_height / 2 + 5),
                action=action,
                zorder=3,
                action_upper_bound=np.array(self.config['env_config']['real_action_upper_bound']),
                action_lower_bound=np.array(self.config['env_config']['real_action_lower_bound']),
            )

            # plot value bar
            if episode_data.paths_value_list:
                value = episode_data.paths_value_list[step]
                ref_allowable = episode_data.ref_allowable[step]
                for artist in self.value_bar_artist_list:
                    artist.remove()
                self.value_bar_artist_list = plot_value_bar(
                    ax=ax,
                    xy=(center_x + 28, center_y - screen_height / 2 + 6),
                    value=value,
                    allowable=ref_allowable,
                    min_value=min_value,
                    max_value=max_value,
                    zorder=3,
                )
            elif episode_data.value_list:
                value = episode_data.value_list[step]
                for artist in self.value_bar_artist_list:
                    artist.remove()
                self.value_bar_artist_list = plot_value_bar(
                    ax=ax,
                    xy=(center_x + 28, center_y - screen_height / 2 + 6),
                    value=[value],
                    allowable=[1],
                    min_value=min_value,
                    max_value=max_value,
                    zorder=3,
                )

            writer.grab_frame()
        writer.finish()
        plt.close(fig)
        print('video export success!')

    def update_ego_ref(self, ax, episode_data, step):
        for ref in self.ref_artist_list:
            ref.remove()
        self.ref_artist_list = []
        selected_path_index = episode_data.selected_path_index_list[step]
        for i in range(len(episode_data.reference_list[step])):
            ref = episode_data.reference_list[step][i][:self.REF_POINT_NUM]
            ref_x, ref_y = ref[:, 0], ref[:, 1]
            self.ref_artist_list.append(ax.add_line(Line2D(
                ref_x, ref_y, color=EGO_COLOR if i == selected_path_index else EGO_COLOR_WITH_ALPHA,
                linewidth=self.REF_LINEWIDTH, zorder=101
            )))

    def adjust_lim(self, episode_data, eval_dict, step):
        index_min = max(0, step - 100)
        index_max = min(len(episode_data.time_stamp_list) - 1, step + 1)
        x_lim_min = episode_data.time_stamp_list[index_min]
        x_lim_max = episode_data.time_stamp_list[index_max]
        self.ax1_1.set_xlim(x_lim_min, x_lim_max)
        min_vx = min(eval_dict['vx_list'][index_min:index_max])
        min_ref_v = min(eval_dict['ref_v_list'][index_min:index_max])
        max_vx = max(eval_dict['vx_list'][index_min:index_max])
        max_ref_v = max(eval_dict['ref_v_list'][index_min:index_max])
        self.ax1_1.set_ylim(min(min_vx, min_ref_v) * 0.95,
                            max(max_vx, max_ref_v) * 1.05)
        self.ax1_2.set_ylim(min(eval_dict['vy_list'][index_min:index_max]) * 0.95,
                            max(eval_dict['vy_list'][index_min:index_max]) * 1.05)
        self.ax1_3.set_ylim(min(eval_dict['r_list'][index_min:index_max]) * 0.95,
                            max(eval_dict['r_list'][index_min:index_max]) * 1.05)
        self.ax2_1.set_xlim(x_lim_min, x_lim_max)
        self.ax2_1.set_ylim(min(eval_dict['y_ref_list'][index_min:index_max]) * 0.95,
                            max(eval_dict['y_ref_list'][index_min:index_max]) * 1.05)
        self.ax2_2.set_ylim(min(eval_dict['ref_phi_list'][index_min:index_max]) * 0.95,
                            max(eval_dict['ref_phi_list'][index_min:index_max]) * 1.05)
        self.ax2_3.set_ylim(min(eval_dict['ref_phi_list'][index_min:index_max]) * 0.95,
                            max(eval_dict['ref_phi_list'][index_min:index_max]) * 1.05)
        self.ax2_4.set_ylim(min(eval_dict['rel_ego_y_list'][index_min:index_max]) * 0.95,
                            max(eval_dict['rel_ego_y_list'][index_min:index_max]) * 1.05)
        self.ax3_1.set_xlim(x_lim_min, x_lim_max)
        min_reward = min([min(v[index_min:index_max]) for v in episode_data.reward_info.values()])
        max_reward = max([max(v[index_min:index_max]) for v in episode_data.reward_info.values()])
        self.ax3_1.set_ylim(min_reward * 0.95, max_reward * 1.05)
        self.ax4_1.set_xlim(x_lim_min, x_lim_max)
        self.ax4_1.set_ylim(min(eval_dict['acc_list'][index_min:index_max]) * 0.95,
                            max(eval_dict['acc_list'][index_min:index_max]) * 1.05)
        self.ax4_2.set_ylim(min(eval_dict['steer_list'][index_min:index_max]) * 0.95,
                            max(eval_dict['steer_list'][index_min:index_max]) * 1.05)

    def plot_figures(self, episode_data, fig, gs):
        eval_dict = {
            'vx_list': [x[0, 0].item() for x in episode_data.obs_list],
            'ref_v_list': [(x[0, 0] - x[0, 7 + self.ref_points_num * 4]).item() for x in episode_data.obs_list],
            'vy_list': [x[0, 1].item() for x in episode_data.obs_list],
            'r_list': [x[0, 2].item() for x in episode_data.obs_list],
            'acc_list': [x[0, 5].item() for x in episode_data.obs_list],
            'steer_list': [x[0, 6].item() * 180 / np.pi for x in episode_data.obs_list],
            "y_ref_list": [x[0, 7 + self.ref_points_num].item() for x in episode_data.obs_list],
            "phi_ref_list": [np.arccos(x[0, 7 + self.ref_points_num*2].item()) * 180 / np.pi for x in episode_data.obs_list],
            "ego_phi_list": [x[4] * 180 / np.pi for x in episode_data.ego_state_list],
            "ego_x_list": [x[0] for x in episode_data.ego_state_list],
            "ego_y_list": [x[1] for x in episode_data.ego_state_list],

            'step_list': episode_data.time_stamp_list,
        }
        eval_dict['ref_phi_list'] = np.array(eval_dict['phi_ref_list']) + np.array(eval_dict['ego_phi_list'])
        ego_phi = np.array(eval_dict['ego_phi_list'])*np.pi/180
        ego_x = np.array(eval_dict['ego_x_list'])
        ego_y = np.array(eval_dict['ego_y_list'])
        # convert the future N step ground coord ego x, y, phi to current ego coord ego x, y, phi every N steps
        N = 100
        convert_base_point_indexs = list(range(0, len(eval_dict['ego_x_list']), N))
        for i in convert_base_point_indexs:
            base_phi = np.mean(ego_phi[i:i+N])
            end_idx = min(i+N, len(eval_dict['ego_x_list']))
            ego_x[i:end_idx], ego_y[i:end_idx], ego_phi[i:end_idx] = convert_ground_coord_to_ego_coord(
                ego_x[i:end_idx], ego_y[i:end_idx], ego_phi[i:end_idx],
                ego_x[i], ego_y[i], base_phi
                )
        eval_dict['rel_ego_phi_list'] = ego_phi.tolist()
        eval_dict['rel_ego_x_list'] = ego_x.tolist()
        eval_dict['rel_ego_y_list'] = ego_y.tolist()
        

            

        

        eval_dict["ego_id"] = episode_data.ego_id
        if 'collision_flag' in episode_data.reward_info:
            del episode_data.reward_info['collision_flag']
        # ego vx, vy, yaw rate
        self.ax1_1, self.ax1_2, self.ax1_3 = plot_vx_vy_r(eval_dict, fig, gs)
        # reference delta y and delta phi
        self.ax2_1, self.ax2_2, self.ax2_3, self.ax2_4 = plot_y_ref_phi_ref(eval_dict, fig, gs)
        # all the non-zero reward
        self.ax3_1 = plot_reward(episode_data, eval_dict, fig, gs)
        # real acc, real steer angle
        self.ax4_1, self.ax4_2 = plot_action(eval_dict, fig, gs)
        return eval_dict

    def plot_traffic(self, episode_data, fig, gs):
        ax = fig.add_subplot(gs[:, 0])  # traffic scene plot
        fig.tight_layout(rect=[0, 0.05, 0.85, 0.95])
        fig.subplots_adjust(wspace=0.1, hspace=0.4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        # plot lane lines
        map_path = episode_data.map_path
        map_data_path = f'{map_path}/scene.pkl'
        with open(map_data_path, 'rb') as f:
            scenario_data: ScenarioData = pickle.load(f)
            network = scenario_data.network
        plot_lane_lines(ax, network, zorder=0)
        # plot ego_id info in the left top corner
        ax.text( # ego_id
            0.01, 0.98, f'ego_id: {episode_data.ego_id}', fontsize=10,
            horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes
        )
        return ax



def convert_ground_coord_to_ego_coord(x, y, phi, ego_x, ego_y, ego_phi):
    shift_x, shift_y = shift(x, y, ego_x, ego_y)
    x_ego_coord, y_ego_coord, phi_ego_coord \
        = rotate(shift_x, shift_y, phi, ego_phi)
    return x_ego_coord, y_ego_coord, phi_ego_coord


def shift(orig_x, orig_y, shift_x, shift_y):
    shifted_x = orig_x - shift_x
    shifted_y = orig_y - shift_y
    return shifted_x, shifted_y


def rotate(orig_x, orig_y, orig_phi, rotate_phi):
    rotated_x = orig_x * np.cos(rotate_phi) + orig_y * np.sin(rotate_phi)
    rotated_y = -orig_x * np.sin(rotate_phi) + \
                orig_y * np.cos(rotate_phi)
    rotated_phi = deal_with_phi_rad(orig_phi - rotate_phi)
    return rotated_x, rotated_y, rotated_phi

def deal_with_phi_rad(phi: float):
    return (phi + np.pi) % (2*np.pi) - np.pi