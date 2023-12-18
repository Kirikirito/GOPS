
import os
import pickle
import numpy as np
from typing import List, Tuple, Dict
from itertools import chain
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Rectangle as Rt
from matplotlib.patches import Circle
from shapely.geometry import LineString

from animation_base import AnimationBase, create_veh, remove_veh, update_veh, veh2vis, rad_to_deg

from process_fcd import FCDLog
from static_background import static_view
from utils import get_line_points

from idscene.scenario import ScenarioData
from idscene.utils.geometry_utils import get_line_inclination
from idsim.utils.coordinates_shift import convert_sumo_coord_to_ground_coord
from render_params import veh_length, veh_width, cyl_length, cyl_width, \
    ped_length, ped_width, traffic_light_length, traffic_light_width, \
    sur_face_color, ego_face_color, ref_color_list


def get_traffic_light_index(current_time: float, durations: np.ndarray) -> int:
    cycle_time = durations.sum()
    current_cycle_time = current_time % cycle_time
    if current_cycle_time < 0.01:
        current_cycle_time = cycle_time
    duration_cunsum = durations.cumsum()
    traffic_light_index = int(sum(current_cycle_time > duration_cunsum))
    return traffic_light_index


class AnimationCross(AnimationBase):
    def __init__(self, theme_style, fcd_file, config) -> None:
        super().__init__(theme_style, fcd_file, config)
        self.task_name = "intersection"
        self.text_counter = 0
        self.fig: plt.Figure = None
        self.ax: plt.Axes = None

    def create_static_path_line(self, ax: plt.Axes, line: LineString, line_color: str):
        px, py = get_line_points(line)
        ax.plot(px, py, color=line_color, linewidth=1,
                ls='--', alpha=0.5, zorder=101)

    # def create_veh(self, ax: plt.Axes, xy:Tuple[float, float], phi:float, length:float, width:float, facecolor:str = sur_face_color) -> Rt:
    #     vehPatch = ax.add_patch(Rt([0, 0], length, width, angle=0., \
    #         facecolor=facecolor, edgecolor=None, zorder=103))
    #     update_veh(vehPatch, xy, phi, length, width)
    #     return vehPatch

    def put_text_on_ax(self, text):
        fontsize_coefficient = 1.3
        self.text_counter += 1
        y = 1.0 - (self.text_counter) * 0.03
        try:
            self.text_list.append(self.ax.text(
                0.0, y-0.02, text,
                transform=self.ax.transAxes,
                fontsize=int(self.fig.get_size_inches()[0] * fontsize_coefficient),
                verticalalignment='top'
            ))
        except Exception as e:
            print('Error!')
            print(e)

    def clear_text(self):
        self.text_counter = 0

    def generate_animation(self, episode_data: Dict, save_video_path: str, episode_index: int, fps=10, mode='debug'):
        metadata = dict(title='Demo', artist='Guojian Zhan', comment='idsim')
        writer = FFMpegWriter(fps=fps, metadata=metadata)

        # ------------------ initialization -------------------
        map_path = episode_data.map_path
        map_data_path = f'{map_path}/scene.pkl'
        with open(map_data_path, 'rb') as f:
            scenario_data: ScenarioData = pickle.load(f)
            network = scenario_data.network
            flow = scenario_data.flow
        # create static background
        fig, ax = static_view(network, self.theme_style)

        # create static path
        ego_route = episode_data.ego_route
        start_lanes = []
        for i in range(len(ego_route) - 1):
            from_lanes = network.get_edge_lanes(ego_route[i])
            to_lanes = network.get_edge_lanes(ego_route[i + 1])
            for from_lane in from_lanes:
                index = 0
                for to_lane in to_lanes:
                    if network.has_connection_line(from_lane, to_lane):
                        if i == 0:
                            start_lanes.append(from_lane)
                        self.create_static_path_line(ax, network.get_lane_center_line(to_lane),
                                                     line_color=ref_color_list[index])
                        self.create_static_path_line(ax, network.get_connection_line(from_lane, to_lane),
                                                     line_color=ref_color_list[index])
                        index += 1
        for start_lane in start_lanes:
            self.create_static_path_line(ax, network.get_lane_center_line(start_lane),
                                         line_color=ref_color_list[0])

        # create ego veh and head
        ego_veh = create_veh(ax, (0, 0), 0, veh_length,
                             veh_width, facecolor=ego_face_color, zorder=103)
        ego_veh_id = episode_data.ego_id
        ego_head = ax.plot([0, 0+veh_length*0.8*np.cos(0)], [0, 0+veh_width *
                           0.8*np.sin(0)], color=ego_face_color, linewidth=0.5, zorder=200)

        ref_lines = []

        # create ego_detect range
        detect_range = 40.0
        ego_detect_circle = ax.add_patch(Circle(
            (0, 0), detect_range, facecolor='gray', edgecolor='black', linewidth=0, alpha=0.2, zorder=200))

        # create traffic light
        # list_list: several lanes of several signalized junctions
        tls_list_list = []
        lane_list_list = []
        lane_Rt_dict: Dict[str:Rt] = {}

        TLS_PHASE_MAPPING = network.TLS_PHASE_MAPPING
        COLOR_MAPPING = {TLS_PHASE_MAPPING['g']: '#00ff00',
                         TLS_PHASE_MAPPING['G']: '#00ff00',
                         TLS_PHASE_MAPPING['y']: '#ffff00',
                         TLS_PHASE_MAPPING['r']: '#ff0000'}
        signalized_junctions = [
            junction for junction in network.nodes if junction.getType() == 'traffic_light']

        for j, junction in enumerate(signalized_junctions):
            connection_list = [junction.getConnections(incoming, outgoing) for incoming in junction.getIncoming() if incoming.getFunction() == ""
                               for outgoing in junction.getOutgoing() if incoming.getFunction() == ""]
            connection_list = list(chain.from_iterable(connection_list))
            connection_list = [connection for connection in connection_list if network._is_connection_allowed(
                connection, v_class='passenger')]
            tls_list = [(connection.getTLSID(), connection.getTLLinkIndex())
                        for connection in connection_list]
            tls_list_list.append(tls_list)
            durations, phase_mat = network._tls[tls_list[0][0]]
            traffic_state_list = [phase_mat[0, tls[1]] for tls in tls_list]

            lane_list = []
            for i in range(len(connection_list)):
                connection = connection_list[i]
                traffic_state = traffic_state_list[i]
                light_color = COLOR_MAPPING[traffic_state]
                lane_id = connection.getFromLane().getID()
                lane_list.append(lane_id)
                lane_line_points_list = list(
                    network.get_lane_center_line(lane_id).coords)
                lane_angle = get_line_inclination(
                    lane_line_points_list[-1], lane_line_points_list[-2])
                traffic_light_xy = lane_line_points_list[-1]
                traffic_light_xy = veh2vis(
                    traffic_light_xy, lane_angle, traffic_light_length, traffic_light_width)

                lane_Rt_dict[lane_id] = ax.add_patch(Rt(traffic_light_xy, traffic_light_length,
                                                        traffic_light_width, angle=rad_to_deg(lane_angle), facecolor=light_color, edgecolor=None, zorder=102))
            lane_list_list.append(lane_list)

        writer.setup(fig, os.path.join(save_video_path,
                     f'{self.task_name}_{episode_index}.mp4'))

        self.text_list = []
        # ---------------------- update-----------------------
        last_veh_id_list = []
        last_cyl_id_list = []
        last_ped_id_list = []
        p_dict = {}
        p_head_dict = {}
        surr_list = []
        for step in range(len(episode_data.time_stamp_list)):
            cur_time = episode_data.time_stamp_list[step]
            cur_time = round(cur_time * 10) / 10

            # # ---------------- update ego veh------------------
            ego_x, ego_y, _, _, ego_phi, _ = episode_data.ego_state_list[step]
            update_veh(ego_veh, (ego_x, ego_y), ego_phi, veh_length, veh_width)
            ego_head[0].set_data([ego_x, ego_x+veh_length*0.8*np.cos(ego_phi)],
                                 [ego_y, ego_y+veh_length*0.8*np.sin(ego_phi)])

            # # ---------------- update ego detect range------------------
            ego_detect_circle.center = (ego_x, ego_y)

            # # ---------------- update ego ref------------------
            for ref in ref_lines:
                ref[0].remove()
            ref_lines = []
            references = episode_data.reference_list[step]
            optimal_index = episode_data.selected_path_index_list[step]
            for i, ref in enumerate(references):
                ref_x, ref_y = ref[:, 0], ref[:, 1]
                if i == optimal_index:
                    ref_lines.append(
                        ax.plot(ref_x, ref_y, color=ref_color_list[i], linewidth=1, zorder=101))

            # # ---------------- update sur participants------------------
            participants = self.fcd_log.at(cur_time).vehicles
            veh_id_list = [p.id for p in participants if p.id !=
                           ego_veh_id and (p.type == 'v1' or p.type == 'v2')]
            cyl_id_list = [p.id for p in participants if p.id !=
                           ego_veh_id and p.type == 'v3']
            ped_id_list = [p.id for p in participants if p.id !=
                           ego_veh_id and p.type == 'person']

            # veh
            to_add_veh = list(set(veh_id_list) - set(last_veh_id_list))
            to_update_veh = list(set(veh_id_list) - set(to_add_veh))
            to_remove_veh = list(set(last_veh_id_list) - set(veh_id_list))
            # cyl
            to_add_cyl = list(set(cyl_id_list) - set(last_cyl_id_list))
            to_update_cyl = list(set(cyl_id_list) - set(to_add_cyl))
            to_remove_cyl = list(set(last_cyl_id_list) - set(cyl_id_list))
            # ped
            to_add_ped = list(set(ped_id_list) - set(last_ped_id_list))
            to_update_ped = list(set(ped_id_list) - set(to_add_ped))
            to_remove_ped = list(set(last_ped_id_list) - set(ped_id_list))

            to_add = to_add_veh + to_add_cyl + to_add_ped
            to_update = to_update_veh + to_update_cyl + to_update_ped
            to_remove = to_remove_veh + to_remove_cyl + to_remove_ped

            for p in participants:
                if p.type == 'v1' or p.type == 'v2':
                    length, width = veh_length, veh_width
                elif p.type == 'v3':
                    length, width = cyl_length, cyl_width
                elif p.type == 'person':
                    length, width = ped_length, ped_width
                x, y, phi = convert_sumo_coord_to_ground_coord(
                    p.x, p.y, p.angle, length)
                if p.id in to_add:
                    p_dict[p.id] = create_veh(
                        ax, (x, y), phi, length, width, facecolor=sur_face_color, zorder=103)
                    p_head_dict[p.id] = ax.plot([x, x+length*0.8*np.cos(phi)], [
                                                y, y+length*0.8*np.sin(phi)], color=sur_face_color, linewidth=0.5, zorder=200)
                elif p.id in to_update:
                    update_veh(p_dict[p.id], (x, y), phi, length, width,)
                    p_head_dict[p.id][0].set_data(
                        [x, x+length*0.8*np.cos(phi)], [y, y+length*0.8*np.sin(phi)])
                elif p.id == ego_veh_id:
                    pass
                else:
                    print('invalid id')
            for p in to_remove:
                remove_veh(p_dict[p])
                del p_dict[p]
            last_veh_id_list = veh_id_list
            last_cyl_id_list = cyl_id_list
            last_ped_id_list = ped_id_list

            # ---------------- update detect sur participants------------------
            # 'lime'
            for p in surr_list:
                remove_veh(p)
            surr_list = []
            surr_states = episode_data.surr_state_list[step][0]
            for p in surr_states:
                x, y, phi, speed, length, width, mask = p
                if mask == 1:
                    surr_list.append(create_veh(
                        ax, (x, y), phi, length, width, facecolor='lime', zorder=103))

            # # ---------------- update traffic light ------------------
            for j, junction in enumerate(signalized_junctions):
                tls_list = tls_list_list[j]
                durations, phase_mat = network._tls[tls_list[0][0]]
                traffic_light_index = get_traffic_light_index(
                    cur_time, durations)
                for i, lane_id in enumerate(lane_list_list[j]):
                    traffic_state = phase_mat[traffic_light_index,
                                              tls_list[i][1]]
                    light_color = COLOR_MAPPING[traffic_state]
                    lane_Rt_dict[lane_id].set_facecolor(light_color)

            # ---------------- update text ------------------
            if mode == 'debug':
                for text in self.text_list:
                    text.remove()
                self.text_list = []

                self.clear_text()
                self.fig, self.ax = fig, ax
                self.put_text_on_ax(f'episode: {episode_index}, step: {step}')

                ego_position_list = ["{:.4f}".format(i) for i in episode_data["ego_state_list"][step][0:2]] + [
                    "{:.4f}".format(episode_data["ego_state_list"][step][4])]
                ego_position_str = ", ".join(ego_position_list)
                self.put_text_on_ax(f'ego postion: {ego_position_str}')

                ego_speed_list = ["{:.4f}".format(
                    i) for i in episode_data["ego_state_list"][step][2:4]]
                ego_speed_str = ", ".join(ego_speed_list)
                self.put_text_on_ax(f'ego speed: {ego_speed_str}')

                # action
                ego_action_list = ["{:.4f}".format(
                    i) for i in episode_data["action_real_list"][step]]
                ego_action_str = ", ".join(ego_action_list)
                self.put_text_on_ax(f'action: {ego_action_str}')

                ## loss
                self.put_text_on_ax(f'loss pi: {episode_data["loss_policy_list"][step]:.4f}')
                self.put_text_on_ax(f'loss lon: {episode_data["loss_lon_list"][step]:.4f}')
                self.put_text_on_ax(f'loss lat: {episode_data["loss_lat_list"][step]:.4f}')
                self.put_text_on_ax(f'loss phi: {episode_data["loss_phi_list"][step]:.4f}')
                self.put_text_on_ax(f'loss vx: {episode_data["loss_vel_list"][step]:.4f}')
                self.put_text_on_ax(f'loss vy: {episode_data["loss_vel_lat_list"][step]:.4f}')
                self.put_text_on_ax(f'loss yaw rate: {episode_data["loss_yaw_rate_list"][step]:.4f}')
                self.put_text_on_ax(f'loss acc: {episode_data["loss_acc_list"][step]:.4f}')
                self.put_text_on_ax(f'loss steer: {episode_data["loss_steer_list"][step]:.4f}')
                self.put_text_on_ax(f'loss acc 1: {episode_data["loss_acc_incremental_list"][step]:.4f}')
                self.put_text_on_ax(f'loss steer 1: {episode_data["loss_steer_incremental_list"][step]:.4f}')
                self.put_text_on_ax(f'loss acc 2: {episode_data["loss_acc_incremental_2nd_list"][step]:.4f}')
                self.put_text_on_ax(f'loss steer 2: {episode_data["loss_steer_incremental_2nd_list"][step]:.4f}')
                self.put_text_on_ax(f'loss c2v: {episode_data["loss_collision2v_list"][step]:.4f}')
                self.put_text_on_ax(f'loss c2v reach: {episode_data["loss_collision2v_reachability_list"][step]:.4f}')
                
                # ## reward
                # text_list.append(
                #     ax.text(-0.1, 0.84, f'env_reward: {episode_data["env_reward_list"][step]:.4f}', transform=ax.transAxes, fontsize=text_fontsize, verticalalignment='top')
                # )

                # text_list.append(
                #     ax.text(-0.1, 0.80, f'paths_loss_policy:', transform=ax.transAxes, fontsize=text_fontsize, verticalalignment='top')
                # )
                # paths_loss_policy_list = ["{:.4f}".format(i) for i in episode_data["paths_loss_policy_list"][step]]
                # paths_loss_policy_str = ", ".join(paths_loss_policy_list)
                # text_list.append(
                #     ax.text(-0.1, 0.76, f'{paths_loss_policy_str}', transform=ax.transAxes, fontsize=text_fontsize, verticalalignment='top')
                # )

                # text_list.append(
                #     ax.text(-0.1, 0.72, f'paths_value:', transform=ax.transAxes, fontsize=text_fontsize, verticalalignment='top')
                # )
                # paths_value_list = ["{:.4f}".format(i) for i in episode_data["paths_value_list"][step]]
                # paths_loss_value_str = ", ".join(paths_value_list)
                # text_list.append(
                #     ax.text(-0.1, 0.68, f'{paths_loss_value_str}', transform=ax.transAxes, fontsize=text_fontsize, verticalalignment='top')
                # )
                # text_list.append(
                #     ax.text(-0.1, 0.64, f'optimal_path_index: {episode_data["selected_path_index_list"][step]}', transform=ax.transAxes, fontsize=text_fontsize, verticalalignment='top')
                # )

                # ## eval
                # text_list.append(
                #     ax.text(-0.1, 0.52, f'eva_risk: {episode_data["eva_risk_list"][step]:.4f}', transform=ax.transAxes, fontsize=text_fontsize, verticalalignment='top')
                # )
                # text_list.append(
                #     ax.text(-0.1, 0.48, f'eva_energy: {episode_data["eva_energy_list"][step]:.4f}', transform=ax.transAxes, fontsize=text_fontsize, verticalalignment='top')
                # )
                # text_list.append(
                #     ax.text(-0.1, 0.44, f'eva_comfort: {episode_data["eva_comfort_list"][step]:.4f}', transform=ax.transAxes, fontsize=text_fontsize, verticalalignment='top')
                # )
                # text_list.append(
                #     ax.text(-0.1, 0.40, f'eva_efficiency: {episode_data["eva_efficiency_list"][step]:.4f}', transform=ax.transAxes, fontsize=text_fontsize, verticalalignment='top')
                # )
                # text_list.append(
                #     ax.text(-0.1, 0.36, f'eva_compliance: {episode_data["eva_compliance_list"][step]:.4f}', transform=ax.transAxes, fontsize=text_fontsize, verticalalignment='top')
                # )

                # ## cost
                # text_list.append(
                #     ax.text(-0.1, 0.28, f'cost_vel: {episode_data["cost_vel_list"][step]:.4f}', transform=ax.transAxes, fontsize=text_fontsize, verticalalignment='top')
                # )
                # text_list.append(
                #     ax.text(-0.1, 0.24, f'cost_lat: {episode_data["cost_lat_list"][step]:.4f}', transform=ax.transAxes, fontsize=text_fontsize, verticalalignment='top')
                # )
                # text_list.append(
                #     ax.text(-0.1, 0.20, f'cost_phi: {episode_data["cost_phi_list"][step]:.4f}', transform=ax.transAxes, fontsize=text_fontsize, verticalalignment='top')
                # )
                # text_list.append(
                #     ax.text(-0.1, 0.16, f'cost_acc: {episode_data["cost_acc_list"][step]:.4f}', transform=ax.transAxes, fontsize=text_fontsize, verticalalignment='top')
                # )
                # text_list.append(
                #     ax.text(-0.1, 0.12, f'cost_steer: {episode_data["cost_steer_list"][step]:.4f}', transform=ax.transAxes, fontsize=text_fontsize, verticalalignment='top')
                # )
                # text_list.append(
                #     ax.text(-0.1, 0.08, f'cost_collision2v: {episode_data["cost_collision2v_list"][step]}', transform=ax.transAxes, fontsize=text_fontsize, verticalalignment='top')
                # )
                # text_list.append(
                #     ax.text(-0.1, 0.04, f'collision_flag: {episode_data["collision_flag_list"][step]}', transform=ax.transAxes, fontsize=text_fontsize, verticalalignment='top')
                # )

            # save ax as png
            # fig.savefig('/home/zhanguojian/idsim-render-zxt/test.png')
            # exit(0)

            writer.grab_frame()
        writer.finish()
        plt.close(fig)
        print('video export success!')
