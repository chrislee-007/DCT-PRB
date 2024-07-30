# 匯入所需的模組
import cv2  # OpenCV影像處理函式庫
import matplotlib.pyplot as plt  # 用於繪製圖像
import numpy as np  # 用於數值計算和數組操作
from collections import deque  # 用於建立固定大小的雙向隊列

# 匯入自定義模組
from tracker import matching  # 物件匹配相關功能
from tracker.gmc import GMC  # 用於目標檢測的幾何多尺度相關功能
from tracker.basetrack import BaseTrack, TrackState  # 物件追蹤基類和追蹤狀態類
from tracker.kalman_filter import KalmanFilter  # 卡爾曼濾波器，用於追蹤預測
from fast_reid.fast_reid_interfece import FastReIDInterface  # 快速ReID介面，用於行人再識別

from .networks_ver12 import load_model  # 自定義模型載入函式
import torch  # PyTorch深度學習函式庫




class STrack(BaseTrack):
    # 使用共享的 KalmanFilter 物件，這個 KalmanFilter 用於追蹤預測
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, cls, feat=None, feat_history=50):
        # 初始化方法，用於創建新的 STrack 物件

        # 等待激活的標誌，當物件被激活時會設置為 True
        self.is_activated = False
        # 追蹤框 (Top-Left-Width-Height) 的座標
        self._tlwh = np.asarray(tlwh, dtype=float)
        #self._tlwh = np.asarray(tlwh, dtype=np.float)

        # KalmanFilter 物件，用於預測目標位置
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        # 物件所屬的類別
        self.cls = -1
        # 用於追蹤類別的歷史記錄，以列表形式存儲
        self.cls_hist = []  # (cls id, freq)
        self.update_cls(cls, score)
        # 物件的追蹤分數
        self.score = score
        # 物件追蹤的長度，初始化為0，每追蹤一次會增加
        self.tracklet_len = 0
        # 平滑後的特徵向量
        self.smooth_feat = None
        # 當前的特徵向量
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        # 特徵向量的歷史記錄，以 deque 形式存儲，最大長度為 feat_history
        self.features = deque([], maxlen=feat_history)
        # 用於平滑特徵向量的參數 alpha
        self.alpha = 0.9

    def update_features(self, feat):
        # 更新物件的特徵向量
        # 特徵向量歸一化
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        # 如果平滑特徵向量為空，將其初始化為當前特徵向量
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            # 使用指數移動平均來平滑特徵向量
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        # 將特徵向量加入特徵記錄中
        self.features.append(feat)
        # 歸一化平滑特徵向量
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def update_cls(self, cls, score):
        # 更新物件的類別信息
        if len(self.cls_hist) > 0:
            max_freq = 0
            found = False
            # 檢查類別是否已在歷史記錄中，如果是，則更新該類別的得分
            for c in self.cls_hist:
                if cls == c[0]:
                    c[1] += score
                    found = True
                # 找出歷史記錄中得分最高的類別
                if c[1] > max_freq:
                    max_freq = c[1]
                    self.cls = c[0]
            # 如果類別不在歷史記錄中，則新增該類別到歷史記錄中
            if not found:
                self.cls_hist.append([cls, score])
                self.cls = cls
        else:
            # 如果歷史記錄為空，則直接新增該類別到歷史記錄中
            self.cls_hist.append([cls, score])
            self.cls = cls

    def predict(self):
        # 預測方法，根據卡爾曼濾波器進行目標位置的預測
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0
        # 透過 KalmanFilter 進行目標位置預測
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        # 靜態方法，批量預測多個目標的位置
        if len(stracks) > 0:
            # 構建用於批量預測的數組
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0

            # 透過共享的 KalmanFilter 物件進行多目標位置預測
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        # 靜態方法，通過幾何多尺度轉換對多個目標位置進行批量變換
        if len(stracks) > 0:
            # 構建用於批量變換的數組
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            # 進行幾何多尺度轉換
            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                # 更新目標位置
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        # 啟動方法，開始一個新的追蹤軌跡
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()

        # 初始化 KalmanFilter 並設置目標位置和協方差
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        # 重新啟動方法，通過新的追蹤目標重新啟動一個已有的軌跡

        # 更新目標位置和協方差
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance,
                                                               self.tlwh_to_xywh(new_track.tlwh))
        # 更新特徵向量、追蹤狀態等資訊
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        # 更新類別信息
        self.update_cls(new_track.cls, new_track.score)

    def update(self, new_track, frame_id):
        # 更新方法，用於更新匹配的目標軌跡
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh

        # 使用 KalmanFilter 進行目標位置更新
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_tlwh))

        # 更新特徵向量
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        # 更新追蹤狀態
        self.state = TrackState.Tracked
        self.is_activated = True

        # 更新追蹤分數和類別信息
        self.score = new_track.score
        self.update_cls(new_track.cls, new_track.score)

    @property
    def tlwh(self):
        """獲取目前位置的邊界框格式 `(左上角 x, 左上角 y, 寬度, 高度)`。"""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """將邊界框轉換為格式 `(最小 x, 最小 y, 最大 x, 最大 y)`，即 `(左上角, 右下角)`。"""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        """將邊界框轉換為格式 `(中心 x, 中心 y, 寬度, 高度)`。"""
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """將邊界框轉換為格式 `(中心 x, 中心 y, 寬高比, 高度)`，其中寬高比為 `寬度 / 高度`。"""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """將邊界框轉換為格式 `(中心 x, 中心 y, 寬度, 高度)`。"""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xywh(self):
        # 將邊界框轉換為格式 `(中心 x, 中心 y, 寬度, 高度)`。
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        # 將邊界框轉換為格式 `(左上角 x, 左上角 y, 寬度, 高度)`。
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        # 將邊界框轉換為格式 `(最小 x, 最小 y, 最大 x, 最大 y)`，即 `(左上角, 右下角)`。
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        # 定義類別的字串表示，格式為 'OT_跟蹤ID(開始幀-結束幀)'。
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


def extract_image_patches(image, bboxes):
    # 將輸入的邊界框位置四捨五入並轉換為整數型別
    bboxes = np.round(bboxes).astype(np.int)
    # 使用邊界框位置來從原始影像中擷取圖像區域，並將擷取的區域存儲在一個列表中
    patches = [image[box[1]:box[3], box[0]:box[2], :] for box in bboxes]
    # 返回擷取的圖像區域列表
    return patches


class SMILEtrack(object):
    def __init__(self, args, frame_rate=30):
        # 初始化方法，創建一個 SMILEtrack 物件

        # 跟蹤中的目標軌跡列表
        self.tracked_stracks = []  # type: list[STrack]
        # 遺失的目標軌跡列表
        self.lost_stracks = []  # type: list[STrack]
        # 已移除的目標軌跡列表
        self.removed_stracks = []  # type: list[STrack]
        # 清空 STtrack 的跟蹤 ID 計數器
        BaseTrack.clear_count()
        # 幀 ID 計數器
        self.frame_id = 0
        # 命令行參數
        self.args = args
        # 跟蹤的閾值，用於確定目標是否為新的追蹤軌跡或已遺失的軌跡
        self.track_high_thresh = args.track_high_thresh
        self.track_low_thresh = args.track_low_thresh
        self.new_track_thresh = args.new_track_thresh
        # 跟蹤緩衝大小和最大遺失幀數
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()
        # ReID 模組相關設定
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh

        # 如果啟用 ReID 模組
        if args.with_reid:
            # 加載 FastReID 模型
            self.weight_path = "./pretrained/ver12.pt"
            self.encoder = load_model(self.weight_path)
            self.encoder = self.encoder.cuda()
            self.encoder = self.encoder.eval()

            # 初始化 GMC (Global Multi-Object Tracking Using Geometry and Appearance) 模組
        self.gmc = GMC(method=args.cmc_method, verbose=[args.name, args.ablation])

    def update(self, output_results, img):
        # 更新方法，用於執行目標追蹤的更新操作

        # 增加幀 ID 計數
        self.frame_id += 1

        # 初始化列表，用於存儲更新後的軌跡
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(output_results):
            # 從輸出結果中獲取檢測結果的位置、分數、類別和特徵向量
            bboxes = output_results[:, :4]
            scores = output_results[:, 4]
            classes = output_results[:, 5]
            features = output_results[:, 6:]

            # 移除低於跟蹤閾值的檢測結果
            lowest_inds = scores > self.track_low_thresh
            bboxes = bboxes[lowest_inds]
            scores = scores[lowest_inds]
            classes = classes[lowest_inds]
            features = output_results[lowest_inds]

            # 找出高於跟蹤閾值的檢測結果
            remain_inds = scores > self.args.track_high_thresh
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
            classes_keep = classes[remain_inds]
            features_keep = features[remain_inds]
        else:
            # 如果沒有檢測結果，則初始化為空列表
            bboxes = []
            scores = []
            classes = []
            dets = []
            scores_keep = []
            classes_keep = []

        '''提取特徵向量 '''
        if self.args.with_reid:
            # 從檢測結果中提取特徵向量

            # 計算特徵向量
            patches_det = extract_image_patches(img, dets)
            features = torch.zeros((len(patches_det), 128), dtype=torch.float64)

            for time in range(len(patches_det)):
                patches_det[time] = torch.tensor(patches_det[time]).cuda()
                features[time, :] = self.encoder.inference_forward_fast(patches_det[time].float())

            features_keep = features.cpu().detach().numpy()

        if len(dets) > 0:
            '''Detections'''
            # 如果有檢測結果
            if self.args.with_reid:
                # 如果啟用了 ReID 模組，則使用檢測結果中的特徵向量創建 STrack 物件
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, c, f) for
                              (tlbr, s, c, f) in zip(dets, scores_keep, classes_keep, features_keep)]
            else:
                # 否則只使用位置、分數和類別信息創建 STrack 物件
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, c) for
                              (tlbr, s, c) in zip(dets, scores_keep, classes_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        # 將新檢測到的目標軌跡添加到 tracked_stracks 列表中
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]#所有目標追蹤樣本都在這，準備進行二階段匹配
        for track in self.tracked_stracks:
            if not track.is_activated:
                # 尚未啟動的目標軌跡暫時添加到 unconfirmed 列表中
                unconfirmed.append(track)
            else:
                # 已啟動的目標軌跡添加到 tracked_stracks 列表中
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        # 第一步：使用高分檢測框進行關聯
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # 使用 KalmanFilter 預測目前的目標位置
        STrack.multi_predict(strack_pool)

        # 修正相機運動
        warp = self.gmc.apply(img, dets)
        STrack.multi_gmc(strack_pool, warp)
        STrack.multi_gmc(unconfirmed, warp)

        # 使用高分檢測框進行關聯
        ious_dists = matching.iou_distance(strack_pool, detections)
        ious_dists_mask = (ious_dists > self.proximity_thresh)

        if not self.args.mot20:
            # 如果不是 MOT20 數據集，則將 iou 距離與分數進行融合
            ious_dists = matching.fuse_score(ious_dists, detections)

        if self.args.with_reid:
            # 如果啟用了 ReID 模組
            dists_iou = matching.iou_distance(strack_pool, detections)
            dists_emb = matching.embedding_distance(strack_pool, detections)
            dists_emb = matching.fuse_motion(self.kalman_filter, dists_emb, strack_pool, detections)

            if dists_emb.size != 0:
                # 融合 iou 距離和特徵向量距離，形成最終的距離度量
                dists = matching.gate(dists_iou, dists_emb)
            else:
                dists = dists_iou

            # Popular ReID method (JDE / FairMOT)
            # raw_emb_dists = matching.embedding_distance(strack_pool, detections)
            # dists = matching.fuse_motion(self.kalman_filter, raw_emb_dists, strack_pool, detections)
            # emb_dists = dists

            # IoU making ReID
            # dists = matching.embedding_distance(strack_pool, detections)
            # dists[ious_dists_mask] = 1.0
        else:
            # 如果沒有啟用 ReID 模組，則直接使用 iou 距離作為最終距離度量
            dists = ious_dists

        # 使用線性分配算法進行目標關聯
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            # 從第一次關聯的匹配中獲取匹配對象
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                # 如果目標軌跡是已追蹤狀態，則進行更新
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                # 否則重新啟動目標軌跡
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # 第三步：使用低分檢測框進行第二次關聯
        if len(scores):
            inds_high = scores < self.args.track_high_thresh
            inds_low = scores > self.args.track_low_thresh
            inds_second = np.logical_and(inds_low, inds_high)
            dets_second = bboxes[inds_second]
            scores_second = scores[inds_second]
            classes_second = classes[inds_second]
        else:
            dets_second = []
            scores_second = []
            classes_second = []

        # 將低分檢測框與未追蹤目標進行關聯
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, c) for
                                 (tlbr, s, c) in zip(dets_second, scores_second, classes_second)]
        else:
            detections_second = []

        # 從未追蹤目標列表中選擇僅有一個開始幀的目標軌跡
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]

        # 計算這些目標軌跡與低分檢測框之間的 iou 距離
        dists = matching.iou_distance(r_tracked_stracks, detections_second)

        # 進行第二次關聯，使用線性分配算法
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                # 如果目標軌跡是已追蹤狀態，則進行更新
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                # 否則重新啟動目標軌跡
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            # 標記未追蹤目標軌跡為遺失狀態
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        # 處理未確認目標軌跡，這些軌跡通常只有一個開始幀
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            # 如果不是 MOT20 數據集，則將 iou 距離與檢測分數進行融合
            dists = matching.fuse_score(dists, detections)

        # 進行第二次未追蹤目標的關聯，使用線性分配算法
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            # 更新未追蹤目標軌跡
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            # 標記未追蹤目標軌跡為已移除狀態
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks """
        # 第四步：初始化新的目標軌跡
        # 對於未追蹤的檢測框中的目標，如果其分數高於新軌跡的閾值，則啟動新的軌跡
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue

            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state """
        # 第五步：更新目標軌跡狀態
        # 對於遺失的軌跡，如果其超過了最大遺失幀數，則將其標記為已移除狀態
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        """ Merge """
        # 合併目標軌跡
        # 首先將已追蹤的軌跡列表中的軌跡狀態為已追蹤的保留下來
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        # 然後將已追蹤的軌跡列表和剛啟動的軌跡列表進行合併
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        # 再將已追蹤的軌跡列表和重新尋找的軌跡列表進行合併
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        # 將已遺失的軌跡列表中存在的已追蹤的軌跡進行剔除
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        # 將新遺失的軌跡加入到已遺失的軌跡列表中
        self.lost_stracks.extend(lost_stracks)
        # 將已移除的軌跡列表中存在的已遺失的軌跡進行剔除
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        # 將新移除的軌跡加入到已移除的軌跡列表中
        self.removed_stracks.extend(removed_stracks)
        # 移除目標軌跡列表中的重複軌跡
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # 獲取輸出的目標軌跡列表，其中已激活的軌跡將被保留
        output_stracks = [track for track in self.tracked_stracks]

        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}  # 用於記錄軌跡的存在狀態的字典
    res = []  # 用於存放合併後的軌跡列表
    # 遍歷第一個軌跡列表 tlista
    for t in tlista:
        exists[t.track_id] = 1  # 將軌跡的 track_id 加入 exists 字典，表示該軌跡存在
        res.append(t)  # 將軌跡加入合併後的軌跡列表 res
    # 遍歷第二個軌跡列表 tlistb
    for t in tlistb:
        tid = t.track_id  # 獲取軌跡的 track_id
        if not exists.get(tid, 0):  # 如果該軌跡的 track_id 在 exists 字典中不存在，表示該軌跡是新的軌跡
            exists[tid] = 1  # 將該軌跡的 track_id 加入 exists 字典，表示該軌跡存在
            res.append(t)  # 將軌跡加入合併後的軌跡列表 res
    return res  # 返回合併後的軌跡列表

def sub_stracks(tlista, tlistb):
    stracks = {}  # 用於存放軌跡的字典，key 是 track_id，value 是對應的軌跡
    # 將第一個軌跡列表 tlista 中的軌跡加入 stracks 字典
    for t in tlista:
        stracks[t.track_id] = t
    # 將第二個軌跡列表 tlistb 中的軌跡與 stracks 字典中的軌跡進行比較
    for t in tlistb:
        tid = t.track_id  # 獲取軌跡的 track_id
        if stracks.get(tid, 0):  # 如果該軌跡的 track_id 在 stracks 字典中存在，表示該軌跡在第一個軌跡列表中
            del stracks[tid]  # 從 stracks 字典中刪除該軌跡
    # 將剩下的軌跡轉換成列表並返回
    return list(stracks.values())

def remove_duplicate_stracks(stracksa, stracksb):
    # 計算兩個軌跡列表中的軌跡兩兩之間的 IoU 距離矩陣
    pdist = matching.iou_distance(stracksa, stracksb)
    # 找出 IoU 距離小於 0.15 的軌跡對應的索引
    pairs = np.where(pdist < 0.15)
    # 存放要刪除的軌跡索引的列表
    dupa, dupb = list(), list()
    # 遍歷每一對符合條件的軌跡索引
    for p, q in zip(*pairs):
        # 計算兩個軌跡的持續時間，即起始帧與當前帧的差值
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        # 判斷哪一個軌跡的持續時間較長，將較長的軌跡的索引添加到對應的列表中
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    # 從原始軌跡列表中刪除重複的軌跡，返回刪除重複後的新軌跡列表
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb

