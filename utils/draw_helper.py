import cv2
import numpy as np
import utils.coordinate as coord


def draw_line(img, start_pt, end_pt, base_pose, res, color, thickness):
    pt0 = coord.tf2ego_location(start_pt, base_pose[:2], base_pose[2])
    pt0 = coord.cart2image(pt0, np.array(img.shape), res).astype(int)
    pt1 = coord.tf2ego_location(end_pt, base_pose[:2], base_pose[2])
    pt1 = coord.cart2image(pt1, np.array(img.shape), res).astype(int)
    cv2.line(img, tuple(pt0.tolist()), tuple(pt1.tolist()), color=color, thickness=thickness)

def draw_lanelines(
    image, lanelines, base_pose, res, channel=2, color_bias=20, thickness=1):
    for laneline in lanelines:
        line, color = laneline['geometry'], laneline['color']
        # channel = 2 if laneline['color'] < 128 else 1
        line_sparse = line[::int(len(line) / 25) + 1]
        if line_sparse[-1] != line[-1]: line_sparse += [line[-1]]
        for pt0, pt1 in zip(line_sparse[:-1], line_sparse[1:]):
            draw_line(
                img=image[channel], start_pt=np.array(pt0), end_pt=np.array(pt1),
                base_pose=base_pose, res=res, color=color+color_bias, thickness=thickness)

def draw_crosswalks(
    image, crosswalks, base_pose, res, channel=2, color=255, thickness=2):
    for crosswalk in crosswalks:
        for pt0, pt1 in crosswalk:
            draw_line(
                img=image[channel], start_pt=np.array(pt0), end_pt=np.array(pt1),
                base_pose=base_pose, res=res, color=color, thickness=thickness)

def draw_junctions(
    image, junctions, base_pose, res, channel=2, color=255, thickness=2):
    for junction in junctions:
        for boundary in junction:
            for pt0, pt1 in boundary:
                draw_line(
                img=image[channel], start_pt=np.array(pt0), end_pt=np.array(pt1),
                base_pose=base_pose, res=res, color=color, thickness=thickness)

def draw_reflines(
    image, ref_lines, base_pose, res, channel=2, color_bias=20, thickness=1):
    for ref_line in ref_lines:
        line, color = ref_line['geometry'], ref_line['color']
        for pt0, pt1 in zip(line[:-1], line[1:]):
            draw_line(
                img=image[channel], start_pt=np.array(pt0), end_pt=np.array(pt1),
                base_pose=base_pose, res=res, color=color+color_bias, thickness=thickness)

def draw_objects(img, objects, base_pose,
    wcmap, res, channel, melt_rate, melt_step, melt_grad, nlimit):
    for otype in objects:
        for oid in objects[otype]:
            states = np.array(objects[otype][oid][::-1][:nlimit])  # now -> past && now -> future
            shape_w, color = wcmap[oid.split('*')[2]]
            draw_states(
                img=img, states=states[::-1], base_pose=base_pose,
                shape_w=shape_w, color=color, res=res, channel=channel,
                melt_rate=melt_rate, melt_step=melt_step, melt_grad=melt_grad)

def draw_states(img, states, base_pose,
    shape_w, color, res, channel, melt_rate, melt_step, melt_grad):
    draw_states_cir(
        img=img, states=states, base_pose=base_pose, 
        res=res, shape_w=shape_w, color=color,
        channel=channel, cir_w=-1, line_w=2,
        melt_rate=melt_rate, melt_step=melt_step, melt_grad=melt_grad)

def draw_states_cir(img, states, base_pose, res, shape_w,
    channel, color, cir_w, line_w, melt_rate, melt_step, melt_grad):
    for i, state in enumerate(states):  # draw past -> now && now -> future
        if np.isnan(state).any() or (np.array(state) == np.inf).any():
            continue

        order = len(states) - 1 - i
        radius = int(max([shape_w * melt_rate**min([order, melt_step]), 0]))
        state_color = max([color - order * melt_grad, 0])

        # prepare location and velocity
        loca = coord.tf2ego_location(state[:2], base_pose[:2], base_pose[2])
        pt2d = coord.cart2image(loca, np.array(img[channel].shape), res).astype(int)
        if (np.abs(pt2d) > img[channel].shape).any():
            continue
        vel = np.sqrt(state[3]**2 + state[4]**2) / 20
        
        # draw connecting line behind
        if i < len(states) - 1:
            thickness = max([2*radius, 1])
            draw_line(img=img[channel],
                start_pt=np.array(states[i][:2]), end_pt=np.array(states[i+1][:2]),
                base_pose=base_pose, res=res, color=state_color, thickness=thickness)
            next_vel = np.sqrt(states[i+1][3]**2 + states[i+1][4]**2) / 20
            draw_line(img=img[(channel-1) % 3],
                start_pt=np.array(states[i][:2]), end_pt=np.array(states[i+1][:2]),
                base_pose=base_pose, res=res, thickness=thickness,
                color=int(min([(vel + next_vel)/2*255, 255])))
        
        # draw locations and velocity front
        try:
            cv2.circle(img=img[channel], center=tuple(pt2d),
                radius=radius, color=state_color, thickness=cir_w)
            cv2.circle(img=img[(channel-1) % 3], center=tuple(pt2d),
                radius=radius, color=int(min([vel*255, 255])), thickness=cir_w)
        except OverflowError:
            print(f'[OverflowError] radius={radius}, center={tuple(pt2d)} '
                  f'state_color={state_color}, thickness={cir_w}, state={state[:2]}, base={base_pose[:2]}')