import argparse
import time
from pathlib import Path
import sys
import os
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import pandas as pd
from prb.models.experimental import attempt_load
from prb.utils.datasets import LoadStreams, LoadImages
from prb.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from prb.utils.plots import plot_one_box
from prb.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from tracker.mc_SMILEtrack import SMILEtrack
from tracker.tracking_utils.timer import Timer
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm
from scipy.ndimage import gaussian_filter

import subprocess

sys.path.insert(0, './prb')
sys.path.append('.')

def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'a') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
def write_results_for_AICity(filename, results):
    with open(filename, 'a') as f:
        for result in results:
            line = ','.join(str(v) for v in result) + '\n'
            f.write(line)
def write_results_for_AICity_MOT(filename, results):
    with open(filename, 'a') as f:
        for result in results:
            line = str(result)
            f.write(line)

black_dots = []
human_dots = []
car_dots = []
scale = 0.7

def bbox_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    area = float(box1_area + box2_area - intersection_area)
    iou = intersection_area / area
    return iou

def is_inside_any_polygon(point, polygon_dir='/home/chris007/python/PRB_SM/AICITY_2023_Track5-main/AICITY_2023_Track5-main/polygon_counter'):
    polygon_files = Path(polygon_dir).glob('*.xlsx')
    for polygon_file in polygon_files:
        df = pd.read_excel(polygon_file)
        vertices = [(x, y) for x, y in zip(df['X'], df['Y'])]
        polygon = Polygon(vertices)
        point = Point(point)
        if polygon.contains(point):
            return True
    return False

def draw_all_polygons(im0, polygon_dir='/home/chris007/python/PRB_SM/AICITY_2023_Track5-main/AICITY_2023_Track5-main/polygon_counter'):
    polygon_files = Path(polygon_dir).glob('*.xlsx')
    for polygon_file in polygon_files:
        df = pd.read_excel(polygon_file)
        vertices = [(x, y) for x, y in zip(df['X'], df['Y'])]
        num_vertices = len(vertices)
        if num_vertices > 1:
            for j in range(num_vertices):
                cv2.line(im0, tuple(vertices[j]), tuple(vertices[(j + 1) % num_vertices]), (0, 255, 0), 2)

def detect(save_img=True):
    first_frame = None
    previous_centers = {}
    source, weights, view_img, save_txt, imgsz, trace, draw = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.trace, opt.draw
    save_img = not opt.nosave and not source.endswith('.txt')
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'

    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()

    trajectory = {}

    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(100)]

    tracker = SMILEtrack(opt, frame_rate=30.0)

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        current_ids = set()
        if first_frame is None:
            first_frame = im0s[0].copy() if isinstance(im0s, list) else im0s.copy()

        if isinstance(im0s, list):
            im0s = [cv2.resize(im0, None, fx=scale, fy=scale) for im0 in im0s]
        else:
            im0s = cv2.resize(im0s, None, fx=scale, fy=scale)

        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        results = []
        AICity_MOT_results = []
        AICity_results = []
        PRINT_results = []

        for i, det in enumerate(pred):
            if webcam:
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                draw_all_polygons(im0)

            detections = []
            if len(det):
                boxes = scale_coords(img.shape[2:], det[:, :4], im0.shape)
                boxes = boxes.cpu().numpy()
                detections = det.cpu().numpy()
                detections[:, :4] = boxes

            detections = np.array(detections)
            online_targets = tracker.update(detections, im0)

            output_list = []

            online_tlwhs = []
            online_tlbr = []
            online_ids = []
            online_scores = []
            online_cls = []
            for t in online_targets:
                tlwh = t.tlwh
                tlbr = t.tlbr
                tid = t.track_id
                tcls = t.cls
                center = ((int(tlwh[0]) + int(tlwh[2])) // 2, (int(tlwh[1]) + int(tlwh[3])) // 2)
                current_ids.add(tid)
                if tid in previous_centers:
                    distance = ((center[0] - previous_centers[tid][0]) ** 2 + (center[1] - previous_centers[tid][1]) ** 2) ** 0.5
                    fps = 30
                    speed = distance * fps
                else:
                    speed = 0
                previous_centers[tid] = center

                if opt.hide_labels_name:
                    label = f'{tid}, {int(tcls)}'
                else:
                    label = f'{tid}, {names[int(tcls)]}'
                label += f', {speed:.2f}px/frame'

                color_classes = [3, 4, 5, 6, 7, 8, 9]
                if tcls in color_classes:
                    color = (255, 0, 0)
                elif tcls == 0:
                    color = (0, 0, 255)
                else:
                    color = colors[int(tid) % len(colors)]

                if save_img or view_img:
                    scale_factors = {
                        3: 1.2,
                        4: 1.2,
                        5: 1.2,
                        6: 1.2,
                        7: 1.2,
                        8: 1.2,
                        9: 1.2,
                        0: 2,
                    }
                    scale_factor = scale_factors.get(tcls, 1.0)
                    offset_w = int((tlbr[2] - tlbr[0]) * (scale_factor - 1))
                    offset_h = int((tlbr[3] - tlbr[1]) * (scale_factor - 1))
                    tlbr_offset = [int(tlbr[0] - offset_w), int(tlbr[1] - offset_h), int(tlbr[2] + offset_w), int(tlbr[3] + offset_h)]

                    output_list.append((int(tlbr_offset[0]), int(tlbr_offset[1]), int(tlbr_offset[2]), int(tlbr_offset[3]), int(tid), int(tcls), int(speed)))
                    cv2.putText(im0, f"Frame {frame}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    plot_one_box(tlbr_offset, im0, label=label, color=color, line_thickness=1)

                if tlwh[2] * tlwh[3] > opt.min_box_area:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    online_cls.append(t.cls)
                    online_tlbr.append(tlbr)
                    results.append(f"{i + 1},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n")
                    filename = p.split("//")[-1].split(".")[0]
                    filename = os.path.basename(filename)
                    t.score = 0.99
                    AICity_MOT_results.append(f"{int(filename)},{int(vid_cap.get(cv2.CAP_PROP_POS_FRAMES))},{tid},{int(tlwh[0])},{int(tlwh[1])},{int(tlwh[2])},{int(tlwh[3])},{int(tcls)},{t.score:.6f}\n")
                    center_x = int(tlwh[0] + tlwh[2] / 2)
                    center_y = int(tlwh[1] + tlwh[3] / 2)
                    frame_number = int(vid_cap.get(cv2.CAP_PROP_POS_FRAMES))
                    AICity_results.append([frame_number, int(tcls) + 1, tid, center_x, center_y, int(tlwh[2]), int(tlwh[3])])
                    PRINT_results.append(f"{int(vid_cap.get(cv2.CAP_PROP_POS_FRAMES))},{tid},{int(tcls)},{int(tlwh[0])},{int(tlwh[1])},{int(tlwh[2])},{int(tlwh[3])},{t.score:.6f}")

            p = Path(p)
            save_path = str(save_dir / p.name)
            output_list = np.array(output_list)
            outputs = [output_list]
            if len(outputs[i]) > 0:
                print(PRINT_results)
                for output in outputs[i]:
                    bboxes = output[0:4]
                    id = output[4]
                    cls = output[5]
                    speed = output[6]

                    collision_classes = [3, 4, 5, 6, 8, 9]
                    if cls == 0:
                        cls0center_x = int((bboxes[0] + bboxes[2]) / 2)
                        cls0center_y = int((bboxes[1] + bboxes[3]) / 2)
                        if is_inside_any_polygon((cls0center_x, cls0center_y)):
                            for bbox_other, cls_other, id_other, speed_other in zip(outputs[i][:, 0:4],outputs[i][:, 5], outputs[i][:, 4],outputs[i][:, 6]):
                                if cls_other in collision_classes and bbox_iou(bboxes,bbox_other) > 0.01 and speed_other > 30:
                                    x = int((bboxes[0] + bboxes[2]) / 2)
                                    y = int((bboxes[1] + bboxes[3]) / 2)
                                    black_dots.append((x, y))
                                    cv2.circle(im0, (x, y), 10, (0, 0, 255), -1)
                                    print("危险！ID：" + str(int(id_other)))
                    for dot in black_dots:
                        cv2.circle(im0, dot, 10, (0, 255, 255), -1)
                    if draw:
                        center = ((int(bboxes[0]) + int(bboxes[2])) // 2, (int(bboxes[1]) + int(bboxes[3])) // 2)
                        if id not in trajectory:
                            trajectory[id] = []
                        trajectory[id].append(center)
                        for i1 in range(1, len(trajectory[id])):
                            if trajectory[id][i1 - 1] is None or trajectory[id][i1] is None:
                                continue
                            thickness = 1
                            try:
                                if cls == 3:
                                    cv2.line(im0, trajectory[id][i1 - 1], trajectory[id][i1], (255, 0, 0), thickness)
                                if cls == 0:
                                    cv2.line(im0, trajectory[id][i1 - 1], trajectory[id][i1], (0, 0, 255), thickness)
                            except:
                                pass

            if view_img:
                cv2.imshow('SMILEtrack', im0)
                cv2.waitKey(1)

            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
        write_results_for_AICity(save_dir / 'labels/AI_result.txt', AICity_results)
        write_results_for_AICity_MOT(save_dir / 'labels/AIMOT_result.txt', AICity_MOT_results)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')
    ids_to_remove = [tid for tid in previous_centers if tid not in current_ids]
    for tid in ids_to_remove:
        del previous_centers[tid]

    if first_frame is not None:
        first_frame_resized = cv2.resize(first_frame, None, fx=scale, fy=scale)
        heatmap = np.zeros_like(first_frame_resized[:, :, 0]).astype(float)
        for dot in black_dots:
            heatmap[dot[1], dot[0]] += 1
        heatmap_log = np.log1p(heatmap)
        heatmap_smoothed = gaussian_filter(heatmap_log, sigma=10)
        if np.max(heatmap_smoothed) > 0:
            heatmap_normalized = (heatmap_smoothed / np.max(heatmap_smoothed) * 255).astype(np.uint8)
        else:
            heatmap_normalized = heatmap_smoothed.astype(np.uint8)
        plt.imshow(first_frame_resized)
        plt.imshow(heatmap_normalized, cmap='jet', alpha=0.5, interpolation='nearest', norm=NoNorm())
        plt.axis('off')
        plt.savefig('heatmap_overlay.jpg', bbox_inches='tight', pad_inches=0, dpi=300)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='video/123.mp4', help='source')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.40, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',default=True, help='display results')
    parser.add_argument('--save-txt', action='store_true',default=True, help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int,help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--trace', action='store_true',default=True, help='trace model')
    parser.add_argument('--hide-labels-name', default=False, action='store_true', help='hide labels')
    parser.add_argument('--draw', action='store_true',default=True, help='display object trajectory lines')
    parser.add_argument("--track_high_thresh", type=float, default=0.3, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.05, type=float, help="lowest detection threshold")
    parser.add_argument("--new_track_thresh", default=0.4, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.7, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--fuse-score", dest="mot20", default=False, action="store_true", help="fuse score and iou for association")
    parser.add_argument("--cmc-method", default="sparseOptFlow", type=str, help="cmc method: sparseOptFlow | files (Vidstab GMC) | orb | ecc")
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="with ReID module.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml", type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth", type=str, help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5, help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25, help='threshold for rejecting low appearance similarity reid matches')

    opt = parser.parse_args()

    opt.jde = False
    opt.ablation = False

    print(opt)
    with torch.no_grad():
        if opt.update:
            for opt.weights in ['weights/yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
