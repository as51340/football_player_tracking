import argparse

import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from collections import Counter, defaultdict

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strong_sort'))  # add strong_sort ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import cv2 as cv

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_boxes, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, print_args, check_file)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, get_bounding_box
from trackers.multi_tracker_zoo import create_tracker

from team_classification import TeamClassificator, DBSCANTeamClassificator, KMeansTeamClassificator
from teams import Team

DARK_BLUE_COLOR = "#13265C"
DARK_RED_COLOR = "#FF5733"
YELLOW_COLOR = "#F7DE3A"

bb_colors = [YELLOW_COLOR, DARK_BLUE_COLOR, DARK_RED_COLOR]  # referee, team1, team2

def get_bgr_color(hex_color):
    rgb = hex2rgb(hex_color)
    return (rgb[2], rgb[1], rgb[0])

def hex2rgb(h):  # rgb order (PIL)
    return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

def load_team_classificator(team_classificator_type: str) -> TeamClassificator:
    if team_classificator_type == "kmeans":
        return KMeansTeamClassificator()
    elif team_classificator_type == "dbscan":
        return DBSCANTeamClassificator()

def add_bounding_box_to_image(cls, id, color, hide_labels, names, hide_conf, hide_class, conf, annotator, bb_info, save_trajectories, tracking_method, q, tracker_list, im0, i):
    c = int(cls)  # integer class
    id = int(id)  # integer id
    label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
        (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
    
    color_bgr = get_bgr_color(color)
    annotator.box_label(bb_info, label, color=color_bgr)
    if save_trajectories and tracking_method == 'strongsort':
        tracker_list[i].trajectory(im0, q, color=color_bgr)

def handle_save_vid(vid_path, vid_writer, vid_cap, save_path, im0, i):
    if vid_path[i] != save_path:  # new video
        vid_path[i] = save_path
        if isinstance(vid_writer[i], cv2.VideoWriter):
            vid_writer[i].release()  # release previous video writer
        if vid_cap:  # video
            fps = vid_cap.get(cv2.CAP_PROP_FPS)
            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:  # stream
            fps, w, h = 30, im0.shape[1], im0.shape[0]
        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    vid_writer[i].write(im0)

@torch.no_grad()
def run(
        source='0',
        yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
        reid_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        tracking_method='strongsort',
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        save_crop=False,  # save cropped prediction boxes
        save_trajectories=False,  # save trajectories for each track
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        team_classificator="kmeans"
):
    
    team_classificator = load_team_classificator(team_classificator_type=team_classificator)

    source = str(source)
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = yolo_weights.stem
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + reid_weights.stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    (save_dir / 'tracks').mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_weights, device=device, dnn=dnn, data=None, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # Create as many strong sort instances as there are video sources
    tracker_list = []
    for i in range(nr_sources):
        tracker = create_tracker(tracking_method, reid_weights, device, half)
        tracker_list.append(tracker, )
        if hasattr(tracker_list[i], 'model'):
            if hasattr(tracker_list[i].model, 'warmup'):
                tracker_list[i].model.warmup()
    outputs = [None] * nr_sources

    # Run tracking
    #model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
    
    
    team1, team2 = None, None
    
    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3
        
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            # video file
            if source.endswith(VID_FORMATS):
                txt_file_name = p.stem
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            # folder with imgs
            else:
                txt_file_name = p.parent.name  # get folder name containing current img
                save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
            curr_frames[i] = im0

            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop

            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            
            if hasattr(tracker_list[i], 'tracker') and hasattr(tracker_list[i].tracker, 'camera_update'):
                if prev_frames[i] is not None and curr_frames[i] is not None:  # camera motion compensation
                    tracker_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # xyxy

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # pass detections to strongsort
                t4 = time_sync()
                outputs[i] = tracker_list[i].update(det.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4
                
                bboxes = []
                max_height, max_width = -1, -1
                
                # mot info
                frame_ids, ids, bbox_lefts = [], [], []
                bbox_tops, bbox_ws, bbox_hs = [], [], []
                clss, confs, qs = [], [], []
                bb_infos = []
                if len(outputs[i]) > 0: 
                    for output in outputs[i]:
                        # Extract bb info
                        bb_info = np.array(output[0:4], dtype=np.int16)
                        id = output[4]
                        cls = output[5]
                        conf = output[6]
                        # Extract info
                        bbox_left = bb_info[0]
                        bbox_top = bb_info[1]
                        bbox_w = bb_info[2] - bb_info[0]
                        bbox_h = bb_info[3] - bb_info[1]
                        bbox = imc[bbox_top:bbox_top+bbox_h, bbox_left:bbox_left+bbox_w, :]
                        bboxes.append(bbox)
                        # Check max coordinates
                        if bbox_h > max_height:
                            max_height = bbox_h
                        if bbox_w > max_width:
                            max_width = bbox_w
                        
                        frame_ids.append(frame_idx+1)
                        ids.append(id)
                        bbox_lefts.append(bbox_left)
                        bbox_tops.append(bbox_top)
                        bbox_ws.append(bbox_w)
                        bbox_hs.append(bbox_h)
                        clss.append(cls)
                        confs.append(conf)
                        qs.append(output[7])
                        bb_infos.append(bb_info)
    
                    # Pad all boxes 
                    padded_bboxes = team_classificator.pad_detections(bboxes, max_width, max_height)
                    labels = team_classificator.classify_persons_into_categories(padded_bboxes, max_height, max_width)
                    assert len(labels) == len(frame_ids)
                    
                    sorted_labels = list(dict(sorted(Counter(labels).items(), key=lambda item: item[1])).keys())
                    # Initialize teams
                    mapping = dict()
                    mapping[sorted_labels[0]] = (sorted_labels[0], YELLOW_COLOR)
                    founded_correct = True
                    if team1 is None or team2 is None:
                        team1 = Team()
                        team1.color = DARK_BLUE_COLOR
                        team1.label = sorted_labels[1]
                        team2 = Team()
                        team2.color = DARK_RED_COLOR
                        team2.label = sorted_labels[2]
                    else:  # there are already some players             
                        k = 0
                        founded_correct = False
                        while labels[k] == sorted_labels[0]:  # find me the player that is not of the smallest class type
                            if (labels[k] == team1.label and ids[k] in team1.players and ids[k] not in team2.players) or (
                                labels[k] == team2.label and ids[k] in team2.players and ids[k] not in team1.players
                            ):
                                founded_correct = True
                                break                        
                            k += 1
                    if founded_correct:
                        mapping[sorted_labels[1]] = (team1.label, team1.color)
                        mapping[sorted_labels[2]] = (team2.label, team2.color)
                    else:
                        mapping[sorted_labels[1]] = (team2.label, team2.color)
                        mapping[sorted_labels[2]] = (team1.label, team1.color)
                                                
                    # Output to MOT format
                    with open(txt_path + '.txt', 'a') as f:
                        for j in range(len(frame_ids)):
                            add_bounding_box_to_image(clss[j], ids[j], mapping[labels[j]][1] , hide_labels, names, hide_conf, hide_class, confs[j], annotator, bb_infos[j], 
                                                      save_trajectories, tracking_method, qs[j], tracker_list, im0, i)
                            f.write(('%g ' * 11 + '\n') % (frame_ids[j], ids[j], mapping[labels[j]][0], bbox_lefts[j],  # MOT format
                                                       bbox_tops[j], bbox_ws[j], bbox_hs[j], -1, -1, -1, i))
                            if mapping[labels[j]][0] == team1.label:
                                team1.players.add(ids[j])
                            elif mapping[labels[j]][0] == team2.label:
                                team2.players.add(ids[j])
                            
                LOGGER.info(f'{s}Done. yolo:({t3 - t2:.3f}s), {tracking_method}:({t5 - t4:.3f}s)')
                
            else:
                LOGGER.info('No detections')
                #tracker_list[i].tracker.pred_n_update_all_tracks()
                
            # Stream results
            im0 = annotator.result()

            # Save results (image with detections)
            handle_save_vid(vid_path, vid_writer, vid_cap, save_path, im0, i)

            prev_frames[i] = curr_frames[i]

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms {tracking_method} update per image at shape {(1, 3, *imgsz)}' % t)
    s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}"
    LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=Path, default=WEIGHTS / 'yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--reid-weights', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--tracking-method', type=str, default='strongsort', help='strongsort, ocsort, bytetrack')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-trajectories', action='store_true', help='save trajectories for each track')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument("--team-classificator", default="kmeans", help="Use clustering algorithm to detect players' teams and referee.")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)