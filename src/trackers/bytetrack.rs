use std::ops::{Deref, DerefMut};

use crate::trackers::Sort;
use ndarray::prelude::*;

/// ByteTrack is a simple tracker that uses the SORT algorithm to track objects.
pub struct ByteTrack {
    pub high_score_threshold: f32,
    pub low_score_threshold: f32,

    sort_tracker: Sort,
}

impl ByteTrack {
    /// Parameters
    /// ----------
    /// * `max_age` - maximum frames a tracklet is kept alive without matching detections
    /// * `min_hits` - minimum number of successive detections before a tracklet is set to alive
    /// * `iou_threshold` - minimum IOU to assign detection to tracklet
    /// * `init_tracker_min_score` - minimum score to create a new tracklet from unmatched detection box
    /// * `high_score_threshold` - boxes with higher scores than this will be used in the first round of association
    /// * `low_score_threshold` - boxes with score between low_score_threshold and high_score_threshold will be used in the second round of association
    /// * `measurement_noise` - Diagonal of the measurement noise covariance matrix,
    ///     i.e. uncertainties of (x, y, s, r) measurements.
    ///     Defaults should be reasonable in most cases
    /// * `process_noise` - Diagonal of the process noise covariance matrix,
    ///     i.e. uncertainties of (x, y, s, r, dx, dy, ds) during each step.
    ///     defaults should be reasonable in most cases
    ///
    /// defaults
    /// * `max_age`: 1
    /// * `min_hits`: 3
    /// * `iou_threshold`: 0.3
    /// * `init_tracker_min_score`: 0.8
    /// * `high_score_threshold`: 0.7
    /// * `low_score_threshold`: 0.1
    /// * `measurement_noise`: \[1., 1., 10., 10.\]
    /// * `process_noise`: \[1., 1., 1., 1., 0.01, 0.01, 0.0001\]
    pub fn new(
        max_age: Option<u32>,
        min_hits: Option<u32>,
        iou_threshold: Option<f32>,
        init_tracker_min_score: Option<f32>,
        high_score_threshold: Option<f32>,
        low_score_threshold: Option<f32>,
        measurement_noise: Option<[f32; 4]>,
        process_noise: Option<[f32; 7]>,
    ) -> Self {
        let sort_tracker = Sort::new(
            max_age,
            min_hits,
            iou_threshold,
            init_tracker_min_score,
            measurement_noise,
            process_noise,
        );

        let high_score_threshold = high_score_threshold.unwrap_or(0.7);
        let low_score_threshold = low_score_threshold.unwrap_or(0.1);

        ByteTrack {
            high_score_threshold,
            low_score_threshold,
            sort_tracker,
        }
    }

    fn split_detections(&self, detection_boxes: CowArray<f32, Ix2>) -> (Array2<f32>, Array2<f32>) {
        let mut high_score_data = Vec::new();
        let mut low_score_data = Vec::new();

        for box_row in detection_boxes.outer_iter() {
            let score = box_row[4];
            if score < self.low_score_threshold {
                continue;
            };
            if score > self.high_score_threshold {
                high_score_data.extend(box_row);
            } else {
                low_score_data.extend(box_row);
            }
        }
        (
            Array2::from_shape_vec((high_score_data.len() / 5, 5), high_score_data).unwrap(),
            Array2::from_shape_vec((low_score_data.len() / 5, 5), low_score_data).unwrap(),
        )
    }

    /// Update the tracker with new boxes and return position of current tracklets
    ///
    /// Parameters
    /// ----------
    /// * `boxes` - array of boxes of shape (n_boxes, 5)
    ///     of the form \[\[xmin1, ymin1, xmax1, ymax1, score1\], \[xmin2,...\],...\]
    /// * `return_all` - if true return all living trackers, including inactive (but not dead) ones
    ///     otherwise return only active trackers (those that got at least min_hits
    ///     matching boxes in a row). Default should be `false`
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
        let tracklet_boxes = self.sort_tracker.predict_and_cleanup();

        let (high_score_detections, low_score_detections) = self.split_detections(detection_boxes);

        let unmatched_high_score_detections = self
            .sort_tracker
            .update_tracklets(high_score_detections.view(), tracklet_boxes.view())?;

        let unmatched_track_box_data: Vec<f32> = tracklet_boxes
            .outer_iter()
            .zip(
                self.sort_tracker
                    .tracklets
                    .iter()
                    .map(|(_, tracker)| tracker.steps_since_update == 0),
            )
            .filter_map(|(box_arr, matched)| if matched { None } else { Some(box_arr) })
            .flatten()
            .copied()
            .collect();
        let unmatched_track_boxes: Array2<f32> = Array2::from_shape_vec(
            (unmatched_track_box_data.len() / 5, 5),
            unmatched_track_box_data,
        )?;

        let unmatched_low_score_detections = self
            .sort_tracker
            .update_tracklets(low_score_detections.view(), unmatched_track_boxes.view())?;

        self.sort_tracker.remove_stale_tracklets();

        self.sort_tracker
            .create_tracklets(unmatched_high_score_detections);
        self.sort_tracker
            .create_tracklets(unmatched_low_score_detections);

        self.sort_tracker.n_steps += 1;
        Ok(self.sort_tracker.get_tracklet_boxes(return_all))
    }

    /// Return current track boxes
    ///
    /// Parameters
    /// ----------
    /// * `return_all` - if true return all living trackers, including inactive (but not dead) ones
    ///     otherwise return only active trackers (those that got at least min_hits
    ///     matching boxes in a row). Default should be `false`.
    ///
    /// Returns
    /// -------
    /// array of tracklet boxes with shape (n_tracks, 5)
    /// of the form \[\[xmin1, ymin1, xmax1, ymax1, track_id1\], \[xmin2,...\],...\]
    pub fn get_current_track_boxes(&self, return_all: bool) -> Array2<f32> {
        self.sort_tracker.get_tracklet_boxes(return_all)
    }
}

// Python inheretance without the python
impl Deref for ByteTrack {
    type Target = Sort;

    fn deref(&self) -> &Self::Target {
        &self.sort_tracker
    }
}

impl DerefMut for ByteTrack {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.sort_tracker
    }
}

impl AsRef<Sort> for ByteTrack {
    fn as_ref(&self) -> &Sort {
        &self.sort_tracker
    }
}

impl AsMut<Sort> for ByteTrack {
    fn as_mut(&mut self) -> &mut Sort {
        &mut self.sort_tracker
    }
}

impl Default for ByteTrack {
    fn default() -> Self {
        Self::new(None, None, None, None, None, None, None, None)
    }
}
