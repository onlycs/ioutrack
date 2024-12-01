use std::collections::BTreeMap;

use ndarray::prelude::*;
use num::cast;

use lapjv::lapjv_rect;

use crate::bbox::{ious, Bbox};
use crate::box_tracker::{KalmanBoxTracker, KalmanBoxTrackerParams};

type TrackidBoxes = Vec<(u32, Bbox<f32>)>;
type ScoreBoxes = Vec<(f32, Bbox<f32>)>;

/// Assign detection boxes to track boxes
///
/// Parameters
/// ----------
/// detections
///     detection boxes, shape (n_detections, 5)
///     of the form [[xmin1, ymin1, xmax1, ymax1, score1], [xmin2,...],...]
/// tracks
///     track boxes, shape (n_tracks, 5)
///     of the form [[xmin1, ymin1, xmax1, ymax1, track_id], [xmin2,...],...]
///
/// Returns
/// -------
/// Tuple of (matches, unmatched_detections)
/// where matches = [(track_id, bbox),...]
/// and unmatched_detections = [(score, bbox),...]
fn assign_detections_to_tracks(
    detections: ArrayView2<f32>,
    tracks: ArrayView2<f32>,
    iou_threshold: f32,
) -> anyhow::Result<(TrackidBoxes, ScoreBoxes)> {
    let mut det_track_ious = ious(detections, tracks);
    det_track_ious.mapv_inplace(|x| -x);
    let (track_idxs, _) = lapjv_rect(det_track_ious.view())?;

    let mut match_updates = Vec::new();
    let mut unmatched_dets = Vec::new();
    for (det_idx, &maybe_track_idx) in track_idxs.iter().enumerate() {
        let det_box = detections.slice(s![det_idx, 0..4]).try_into().unwrap();
        let score = detections[(det_idx, 4)];
        match maybe_track_idx {
            Some(track_idx) => {
                // we negated the ious, so negate again here
                if -det_track_ious[(det_idx, track_idx)] > iou_threshold {
                    match_updates.push((tracks[(track_idx, 4)] as u32, det_box))
                } else {
                    unmatched_dets.push((score, det_box));
                }
            }
            None => unmatched_dets.push((score, det_box)),
        }
    }

    Ok((match_updates, unmatched_dets))
}

pub struct Sort {
    pub max_age: u32,
    pub min_hits: u32,
    pub iou_threshold: f32,
    pub init_tracker_min_score: f32,
    /// id of next tracklet initialized
    next_track_id: u32,
    measurement_noise: [f32; 4],
    process_noise: [f32; 7],
    /// current tracklets
    pub tracklets: BTreeMap<u32, KalmanBoxTracker>,
    /// number of steps the tracker has run for
    pub n_steps: u32,
}

impl Sort {
    /// Create a new SORT bbox tracker
    ///
    /// Parameters
    /// ----------
    /// * `max_age` - maximum frames a tracklet is kept alive without matching detections
    /// * `min_hits` - minimum number of successive detections before a tracklet is set to alive
    /// * `iou_threshold` minimum IOU to assign detection to tracklet
    /// * `init_tracker_min_score` - minimum score to create a new tracklet from unmatched detection box
    /// * `measurement_noise` - Diagonal of the measurement noise covariance matrix,
    ///     i.e. uncertainties of (x, y, s, r) measurements,
    ///     defaults should be reasonable in most cases
    /// * `process_noise` - Diagonal of the process noise covariance matrix,
    ///     i.e. uncertainties of (x, y, s, r, dx, dy, ds) during each step.
    ///     defaults should be reasonable in most cases
    ///
    /// defaults:
    /// * `max_age`: 1
    /// * `min_hits`: 3
    /// * `iou_threshold`: 0.3
    /// * `init_tracker_min_score`: 0.0
    /// * `measurement_noise`: \[1., 1., 10., 0.05\]
    /// * `process_noise`: \[1., 1., 1., 0.001, 0.01, 0.01, 0.0001\]
    pub fn new(
        max_age: Option<u32>,
        min_hits: Option<u32>,
        iou_threshold: Option<f32>,
        init_tracker_min_score: Option<f32>,
        measurement_noise: Option<[f32; 4]>,
        process_noise: Option<[f32; 7]>,
    ) -> Self {
        let max_age = max_age.unwrap_or(1);
        let min_hits = min_hits.unwrap_or(3);
        let iou_threshold = iou_threshold.unwrap_or(0.3);
        let init_tracker_min_score = init_tracker_min_score.unwrap_or(0.0);
        let measurement_noise = measurement_noise.unwrap_or([1., 1., 10., 0.05]);
        let process_noise = process_noise.unwrap_or([1., 1., 1., 0.001, 0.01, 0.01, 0.0001]);

        Sort {
            max_age,
            min_hits,
            iou_threshold,
            init_tracker_min_score,
            measurement_noise,
            process_noise,
            next_track_id: 1,
            tracklets: BTreeMap::new(),
            n_steps: 0,
        }
    }

    pub fn predict_and_cleanup(&mut self) -> Array2<f32> {
        // estimate of capacity assumes that all trackers are valid
        // this should be the case most of the time
        let mut data = Vec::with_capacity(self.tracklets.len() * 5);

        // get predicted boxes,
        // filter out trackers that return invalid boxes
        self.tracklets.retain(|_, tracklet| {
            let b = tracklet.predict();
            let bounds = b.to_bounds();
            if b.xmin >= b.xmax || b.ymin >= b.ymax || bounds.iter().any(|x| !x.is_normal()) {
                return false;
            }
            data.extend(bounds);
            data.push(cast(tracklet.id).unwrap());
            true
        });
        Array2::from_shape_vec((self.tracklets.len(), 5), data).unwrap()
    }

    /// Return current track boxes
    ///
    /// Parameters
    /// ----------
    /// * `return_all` - if true return all living trackers, including inactive
    ///     (but not dead) ones otherwise return only active trackers
    ///     (those that got at least min_hits matching boxes in a row).
    ///      Default is false.
    ///
    /// Returns
    /// -------
    /// array of tracklet boxes with shape (n_tracks, 5)
    /// of the form \[\[xmin1, ymin1, xmax1, ymax1, track_id1\], \[xmin2,...\],...\]
    pub fn get_tracklet_boxes(&self, return_all: bool) -> Array2<f32> {
        let mut data = Vec::new();
        for (_, tracklet) in self.tracklets.iter() {
            if return_all
                || (tracklet.steps_since_update < 1
                    && (tracklet.hit_streak >= self.min_hits || self.n_steps <= self.min_hits))
            {
                data.extend(tracklet.bbox().to_bounds());
                data.push(cast(tracklet.id).unwrap());
            }
        }
        Array2::from_shape_vec((data.len() / 5, 5), data).unwrap()
    }

    pub fn create_tracklets(&mut self, score_boxes: ScoreBoxes) {
        for (score, bbox) in score_boxes {
            if score >= self.init_tracker_min_score {
                self.tracklets.insert(
                    self.next_track_id,
                    KalmanBoxTracker::new(KalmanBoxTrackerParams {
                        id: self.next_track_id,
                        bbox,
                        meas_var: Some(self.measurement_noise),
                        proc_var: Some(self.process_noise),
                    }),
                );
                self.next_track_id += 1
            }
        }
    }

    pub fn update_tracklets(
        &mut self,
        detection_boxes: ArrayView2<f32>,
        tracklet_boxes: ArrayView2<f32>,
    ) -> anyhow::Result<ScoreBoxes> {
        let (matched_boxes, unmatched_detections) =
            assign_detections_to_tracks(detection_boxes, tracklet_boxes, self.iou_threshold)?;

        for (track_id, bbox) in matched_boxes {
            let update_result = self.tracklets.get_mut(&track_id).unwrap().update(bbox);
            if update_result.is_err() {
                // Failed to invert S matrix, broken tracklet
                self.tracklets.remove(&track_id);
            }
        }
        Ok(unmatched_detections)
    }

    pub fn remove_stale_tracklets(&mut self) {
        self.tracklets
            .retain(|_, tracklet| tracklet.steps_since_update <= self.max_age);
    }

    /// Update the tracker with new boxes and return position of current tracklets
    ///
    /// Parameters
    /// ----------
    /// * `boxes` - array of boxes of shape (n_boxes, 5)
    ///     of the form \[\[xmin1, ymin1, xmax1, ymax1, score1\], \[xmin2,...\],...\]
    /// * `return_all` - if true return all living trackers, including inactive
    ///     (but not dead) ones otherwise return only active trackers (those that got at
    ///     least min_hits matching boxes in a row). Default is false.
    ///
    /// Returns
    /// -------
    /// array of tracklet boxes with shape (n_tracks, 5)
    /// of the form \[\[xmin1, ymin1, xmax1, ymax1, track_id1\], \[xmin2,...\],...\]
    pub fn update(
        &mut self,
        detection_boxes: CowArray<f32, Ix2>,
        return_all: bool,
    ) -> anyhow::Result<Array2<f32>> {
        let tracklet_boxes = self.predict_and_cleanup();
        let unmatched_detections =
            self.update_tracklets(detection_boxes.view(), tracklet_boxes.view())?;

        self.remove_stale_tracklets();

        self.create_tracklets(unmatched_detections);

        self.n_steps += 1;
        Ok(self.get_tracklet_boxes(return_all))
    }

    /// Remove all current tracklets
    pub fn clear(&mut self) {
        self.tracklets.clear();
    }

    /// Remove the tracklet with the given track_id
    /// no effect if the track_id is not in use
    pub fn remove_tracker(&mut self, track_id: u32) {
        self.tracklets.remove(&track_id);
    }
}

impl Default for Sort {
    fn default() -> Self {
        Self::new(None, None, None, None, None, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_first_update() {
        let mut tracker = Sort::new(
            Some(1),
            Some(3),
            Some(0.3),
            Some(0.3),
            Some([1., 1., 10., 10.]),
            Some([1., 1., 1., 1., 0.01, 0.01, 0.0001]),
        );

        assert_abs_diff_eq!(
            tracker
                .update(
                    array![[0.0, 1.5, 12.6, 25.0, 0.9], [-5.5, 18.0, 1.0, 20.0, 0.15]].into(),
                    false
                )
                .unwrap(),
            array![[0.0, 1.5, 12.6, 25.0, 1.0]],
            epsilon = 0.00001
        )
    }

    #[test]
    fn test_filter() {
        let mut tracker = Sort::new(
            Some(1),
            Some(3),
            Some(0.3),
            Some(0.3),
            Some([1., 1., 10., 10.]),
            Some([1., 1., 1., 1., 0.01, 0.01, 0.0001]),
        );

        tracker
            .update(
                array![
                    [f32::INFINITY, 1.5, 12.6, 25.0, 0.9],
                    [-5.5, 18.0, 1.0, 20.0, 0.15]
                ]
                .into(),
                false,
            )
            .unwrap();
        let res = tracker.predict_and_cleanup();
        assert!(res.shape()[0] == 0);
    }
}
