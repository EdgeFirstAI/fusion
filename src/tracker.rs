use crate::kalman::ConstantVelocityXYAHModel2;
use lapjv::{lapjv, Matrix};
use log::{debug, trace};
use nalgebra::{Dyn, OMatrix, U4};
use std::collections::HashMap;
use uuid::Uuid;

pub struct ByteTrackSettings {
    pub track_high_conf: f32,
    pub track_extra_lifespan: f32,
    pub track_iou: f32,
    pub track_update: f32,
}
#[allow(dead_code)]
pub struct ByteTrack {
    // tracklets;
    pub tracklets: Vec<Tracklet>,
    pub lost_tracks: Vec<Tracklet>,
    pub removed_tracks: Vec<Tracklet>,
    pub frame_count: i32,
    pub timestamp: u64,
    pub uuid_map_vision_class: HashMap<Uuid, u8>,
    pub uuid_map_fusion_class: HashMap<Uuid, u8>,
    pub settings: ByteTrackSettings,
}
#[derive(Debug, Clone)]
pub struct Tracklet {
    pub id: Uuid,
    pub prev_boxes: TrackerBox,
    pub filter: ConstantVelocityXYAHModel2<f32>,
    pub expiry: u64,
    pub last_updated: u64,
    pub last_updated_high_conf: u64,
    pub count: i32,
    pub created: u64,
}
#[derive(Debug, Clone, Copy)]
pub struct TrackerBox {
    pub xmin: f32,
    pub ymin: f32,
    pub xmax: f32,
    pub ymax: f32,
    pub score: f32,
    pub vision_class: u8,
    pub fusion_class: u8,
}

impl Tracklet {
    fn update(&mut self, vaalbox: &TrackerBox, s: &ByteTrackSettings, ts: u64) {
        self.count += 1;
        self.expiry = ts + (s.track_extra_lifespan * 1e9) as u64;
        self.last_updated = ts;
        if vaalbox.score >= s.track_high_conf {
            self.last_updated_high_conf = ts;
        }
        self.prev_boxes = *vaalbox;
        self.filter.update(&vaalbox_to_xyah(vaalbox));
    }

    pub fn get_predicted_location(&self) -> TrackerBox {
        let predicted_xyah = self.filter.mean.as_slice();
        let mut expected = TrackerBox {
            xmin: 0.0,
            xmax: 0.0,
            ymin: 0.0,
            ymax: 0.0,
            score: self.prev_boxes.score,
            vision_class: self.prev_boxes.vision_class,

            fusion_class: self.prev_boxes.fusion_class,
        };
        xyah_to_vaalbox(predicted_xyah, &mut expected);
        expected
    }
}

fn vaalbox_to_xyah(vaal_box: &TrackerBox) -> [f32; 4] {
    let x = (vaal_box.xmax + vaal_box.xmin) / 2.0;
    let y = (vaal_box.ymax + vaal_box.ymin) / 2.0;
    let w = (vaal_box.xmax - vaal_box.xmin).max(EPSILON);
    let h = (vaal_box.ymax - vaal_box.ymin).max(EPSILON);
    let a = w / h;

    [x, y, a, h]
}

fn xyah_to_vaalbox(xyah: &[f32], vaal_box: &mut TrackerBox) {
    if xyah.len() < 4 {
        return;
    }
    let x_ = xyah[0];
    let y_ = xyah[1];
    let a_ = xyah[2];
    let h_ = xyah[3];
    let w_ = h_ * a_;
    vaal_box.xmin = x_ - w_ / 2.0;
    vaal_box.xmax = x_ + w_ / 2.0;
    vaal_box.ymin = y_ - h_ / 2.0;
    vaal_box.ymax = y_ + h_ / 2.0;
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct TrackInfo {
    pub uuid: Uuid,
    pub count: i32,
    pub created: u64,
}
const INVALID_MATCH: f32 = 1000000.0;
const EPSILON: f32 = 0.00001;

fn iou(box1: &TrackerBox, box2: &TrackerBox) -> f32 {
    let intersection = (box1.xmax.min(box2.xmax) - box1.xmin.max(box2.xmin)).max(0.0)
        * (box1.ymax.min(box2.ymax) - box1.ymin.max(box2.ymin)).max(0.0);

    if intersection <= EPSILON {
        return 0.0;
    }

    let union = (box1.xmax - box1.xmin) * (box1.ymax - box1.ymin)
        + (box2.xmax - box2.xmin) * (box2.ymax - box2.ymin)
        - intersection;

    if union <= EPSILON {
        return 0.0;
    }

    intersection / union
}

fn box_cost(
    track: &Tracklet,
    new_box: &TrackerBox,
    distance: f32,
    score_threshold: f32,
    iou_threshold: f32,
) -> f32 {
    let _ = distance;

    if new_box.score < score_threshold {
        return INVALID_MATCH;
    }

    // use iou between predicted box and real box:
    let predicted_xyah = track.filter.mean.as_slice();
    let mut expected = TrackerBox {
        xmin: 0.0,
        xmax: 0.0,
        ymin: 0.0,
        ymax: 0.0,
        score: 0.0,
        vision_class: 0,

        fusion_class: 0,
    };
    xyah_to_vaalbox(predicted_xyah, &mut expected);
    let iou = iou(&expected, new_box);
    if iou < iou_threshold {
        return INVALID_MATCH;
    }
    (1.5 - new_box.score) + (1.5 - iou)
}

impl ByteTrack {
    #[allow(dead_code)]
    pub fn new() -> ByteTrack {
        ByteTrack {
            tracklets: vec![],
            lost_tracks: vec![],
            removed_tracks: vec![],
            frame_count: 0,
            timestamp: 0,
            uuid_map_vision_class: HashMap::new(),

            uuid_map_fusion_class: HashMap::new(),
            settings: ByteTrackSettings {
                track_extra_lifespan: 0.5,
                track_high_conf: 0.5,
                track_iou: 0.1,
                track_update: 0.4,
            },
        }
    }

    pub fn new_with_settings(settings: ByteTrackSettings) -> ByteTrack {
        ByteTrack {
            tracklets: vec![],
            lost_tracks: vec![],
            removed_tracks: vec![],
            frame_count: 0,
            timestamp: 0,
            uuid_map_vision_class: HashMap::new(),
            uuid_map_fusion_class: HashMap::new(),
            settings,
        }
    }

    fn compute_costs(
        &mut self,
        boxes: &[TrackerBox],
        score_threshold: f32,
        iou_threshold: f32,
        box_filter: &[bool],
        track_filter: &[bool],
    ) -> Matrix<f32> {
        // costs matrix must be square
        let dims = boxes.len().max(self.tracklets.len());
        let mut measurements = OMatrix::<f32, Dyn, U4>::from_element(boxes.len(), 0.0);
        for (i, mut row) in measurements.row_iter_mut().enumerate() {
            row.copy_from_slice(&vaalbox_to_xyah(&boxes[i]));
        }

        // TODO: use matrix math for IOU, should speed up computation, and store it in
        // distances

        Matrix::from_shape_fn((dims, dims), |(x, y)| {
            if x < boxes.len() && y < self.tracklets.len() {
                if box_filter[x] || track_filter[y] {
                    INVALID_MATCH
                } else {
                    box_cost(
                        &self.tracklets[y],
                        &boxes[x],
                        // distances[(x, y)],
                        0.0,
                        score_threshold,
                        iou_threshold,
                    )
                }
            } else {
                0.0
            }
        })
    }

    fn match_tracklets_low_score(
        &mut self,
        boxes: &mut [TrackerBox],
        matched: &mut [bool],
        tracked: &mut [bool],
        matched_info: &mut [Option<TrackInfo>],
        timestamp: u64,
    ) {
        // try to match unmatched tracklets to low score detections as well
        if !self.tracklets.is_empty() {
            let costs = self.compute_costs(boxes, 0.0, self.settings.track_iou, matched, tracked);
            let ans = lapjv(&costs).unwrap();
            for i in 0..ans.0.len() {
                let x = ans.0[i];
                if i < boxes.len() && x < self.tracklets.len() {
                    // matched tracks
                    // We need to filter out those "invalid" assignments
                    if matched[i] || tracked[x] || (costs[(i, x)] >= INVALID_MATCH) {
                        continue;
                    }
                    matched[i] = true;
                    matched_info[i] = Some(TrackInfo {
                        uuid: self.tracklets[x].id,
                        count: self.tracklets[x].count,
                        created: self.tracklets[x].created,
                    });
                    trace!(
                        "Cost: {} Box: {:#?} UUID: {} Mean: {}",
                        costs[(i, x)],
                        boxes[i],
                        self.tracklets[x].id,
                        self.tracklets[x].filter.mean
                    );
                    assert!(!tracked[x]);
                    tracked[x] = true;
                    let predicted_xyah = self.tracklets[x].filter.mean.as_slice();
                    let x_ = predicted_xyah[0];
                    let y_ = predicted_xyah[1];
                    let a_ = predicted_xyah[2];
                    let h_ = predicted_xyah[3];

                    self.tracklets[x].update(&boxes[i], &self.settings, timestamp);

                    let w_ = h_ * a_;
                    boxes[i].xmin = x_ - w_ / 2.0;
                    boxes[i].xmax = x_ + w_ / 2.0;
                    boxes[i].ymin = y_ - h_ / 2.0;
                    boxes[i].ymax = y_ + h_ / 2.0;
                }
            }
        }
    }

    fn remove_expired_tracklets(&mut self, timestamp: u64) {
        for i in (0..self.tracklets.len()).rev() {
            if self.tracklets[i].expiry < timestamp {
                debug!("Tracklet removed: {:?}", self.tracklets[i].id);
                self.uuid_map_vision_class
                    .remove_entry(&self.tracklets[i].id);

                self.uuid_map_fusion_class
                    .remove_entry(&self.tracklets[i].id);
                let _ = self.tracklets.swap_remove(i);
            }
        }
    }

    fn create_new_tracklets_from_high_score(
        &mut self,
        boxes: &[TrackerBox],
        high_conf_ind: Vec<usize>,
        matched: &[bool],
        matched_info: &mut [Option<TrackInfo>],
        timestamp: u64,
    ) {
        for i in high_conf_ind {
            if !matched[i] {
                let id = Uuid::new_v4();
                matched_info[i] = Some(TrackInfo {
                    uuid: id,
                    count: 1,
                    created: timestamp,
                });
                self.tracklets.push(Tracklet {
                    id,
                    prev_boxes: boxes[i],
                    filter: ConstantVelocityXYAHModel2::new(
                        &vaalbox_to_xyah(&boxes[i]),
                        self.settings.track_update,
                    ),
                    expiry: timestamp + (self.settings.track_extra_lifespan * 1e9) as u64,
                    last_updated: timestamp,
                    last_updated_high_conf: timestamp,
                    count: 1,
                    created: timestamp,
                });
                self.uuid_map_vision_class.insert(id, boxes[i].vision_class);

                self.uuid_map_fusion_class.insert(id, boxes[i].fusion_class);
            }
        }
    }

    pub fn update(&mut self, boxes: &mut [TrackerBox], timestamp: u64) -> Vec<Option<TrackInfo>> {
        self.frame_count += 1;
        let high_conf_ind = (0..boxes.len())
            .filter(|x| boxes[*x].score >= self.settings.track_high_conf)
            .collect::<Vec<usize>>();
        let mut matched = vec![false; boxes.len()];
        let mut tracked = vec![false; self.tracklets.len()];
        let mut matched_info = vec![None; boxes.len()];
        if !self.tracklets.is_empty() {
            for track in &mut self.tracklets {
                track.filter.predict();
            }
            let costs = self.compute_costs(
                boxes,
                self.settings.track_high_conf,
                self.settings.track_iou,
                &matched,
                &tracked,
            );
            // With m boxes and n tracks, we compute a m x n array of costs for
            // association cost is based on distance computed by the Kalman Filter
            // Then we use lapjv (linear assignment) to minimize the cost of
            // matching tracks to boxes
            // The linear assignment will still assign some tracks to out of threshold
            // scores/filtered tracks/filtered boxes But it will try to minimize
            // the number of "invalid" assignments, since those are just very high costs
            let ans = lapjv(&costs).unwrap();
            for i in 0..ans.0.len() {
                let x = ans.0[i];
                if !(i < boxes.len() && x < self.tracklets.len()) {
                    continue;
                }
                // We need to filter out those "invalid" assignments
                if costs[(i, ans.0[i])] >= INVALID_MATCH {
                    continue;
                }
                matched[i] = true;
                matched_info[i] = Some(TrackInfo {
                    uuid: self.tracklets[x].id,
                    count: self.tracklets[x].count,
                    created: self.tracklets[x].created,
                });
                assert!(!tracked[x]);
                tracked[x] = true;

                let observed_box = boxes[i];

                let predicted_xyah = self.tracklets[x].filter.mean.as_slice();
                xyah_to_vaalbox(predicted_xyah, &mut boxes[i]);
                self.uuid_map_vision_class
                    .insert(self.tracklets[x].id, boxes[i].vision_class);

                self.uuid_map_fusion_class
                    .insert(self.tracklets[x].id, boxes[i].fusion_class);
                self.tracklets[x].update(&observed_box, &self.settings, timestamp);
            }
        }

        // try to match unmatched tracklets to low score detections as well
        self.match_tracklets_low_score(
            boxes,
            &mut matched,
            &mut tracked,
            &mut matched_info,
            timestamp,
        );

        // move tracklets that don't have lifespan to the removed tracklets
        // must iterate from the back
        self.remove_expired_tracklets(timestamp);

        // unmatched high score boxes are then used to make new tracks
        self.create_new_tracklets_from_high_score(
            boxes,
            high_conf_ind,
            &matched,
            &mut matched_info,
            timestamp,
        );
        matched_info
    }

    pub fn get_tracklets(&self) -> &Vec<Tracklet> {
        &self.tracklets
    }

    pub fn get_tracklet_from_uuid(&self, uuid: &Uuid) -> Option<&Tracklet> {
        self.tracklets.iter().find(|t| t.id == *uuid)
    }
}

#[cfg(test)]
mod tests {

    use crate::tracker::TrackerBox;

    use super::{vaalbox_to_xyah, xyah_to_vaalbox};

    #[test]
    fn filter() {
        let box1 = TrackerBox {
            xmin: 0.02135,
            xmax: 0.12438,
            ymin: 0.0134,
            ymax: 0.691,
            score: 0.0,
            vision_class: 0,

            fusion_class: 0,
        };
        let xyah = vaalbox_to_xyah(&box1);
        let mut box2 = TrackerBox {
            xmin: 0.0,
            xmax: 0.0,
            ymin: 0.0,
            ymax: 0.0,
            score: 0.0,
            vision_class: 0,

            fusion_class: 0,
        };
        xyah_to_vaalbox(&xyah, &mut box2);

        assert!((box1.xmax - box2.xmax).abs() < f32::EPSILON);
        assert!((box1.ymax - box2.ymax).abs() < f32::EPSILON);
        assert!((box1.xmin - box2.xmin).abs() < f32::EPSILON);
        assert!((box1.ymin - box2.ymin).abs() < f32::EPSILON);
    }
}
