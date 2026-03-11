// Copyright 2025 Au-Zone Technologies Inc.
// SPDX-License-Identifier: Apache-2.0

use edgefirst_schemas::edgefirst_msgs::Mask;
use itertools::Itertools;
use tracing::instrument;

/// Sentinel value for points not classified by the vision model.
/// Using u8::MAX (255) reserves it as a sentinel; valid class indices are 0..=254.
pub const UNCLASSIFIED: u8 = u8::MAX;

/// Finds the argmax of the slice. Panics if the slice is empty.
/// Indices >= 255 are clamped to UNCLASSIFIED (out-of-range for valid classes).
pub fn argmax_slice<T: Ord>(slice: &[T]) -> u8 {
    slice
        .iter()
        .position_max()
        .unwrap()
        .min(UNCLASSIFIED as usize) as u8
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Box2D {
    pub center_x: f32,
    pub center_y: f32,
    pub width: f32,
    pub height: f32,
    pub label: u8,
}

/// Decompress (if zstd) and argmax a multi-channel mask into a single-channel
/// class-index mask. Modifies the mask in-place. Returns the number of
/// channels detected (1 if single-channel or invalid, >1 if argmax was applied).
pub fn process_mask(mask: &mut Mask) -> usize {
    if mask.encoding == "zstd" {
        match zstd::decode_all(mask.mask.as_slice()) {
            Ok(decoded) => {
                mask.encoding = String::new();
                mask.mask = decoded;
            }
            Err(e) => {
                log::error!("Failed to decompress zstd mask: {e}");
                mask.mask.clear();
                mask.width = 0;
                mask.height = 0;
                return 1;
            }
        }
    }
    if mask.width == 0 || mask.height == 0 {
        mask.mask.clear();
        return 1;
    }
    let pixels = mask.width as usize * mask.height as usize;
    if mask.mask.is_empty() || pixels == 0 || mask.mask.len() % pixels != 0 {
        mask.mask = vec![0; pixels];
        return 1;
    }
    let channels = mask.mask.len() / pixels;
    if channels > 1 {
        mask.mask = mask.mask.chunks_exact(channels).map(argmax_slice).collect();
    }
    channels
}

/// Resolve a box label to a u8 class index using the labels list.
/// Returns `None` if the label is unrecognized or resolves to the reserved
/// UNCLASSIFIED sentinel (255). Valid class indices are 0..=254.
pub fn resolve_box_label(label: &str, labels: Option<&[String]>) -> Option<u8> {
    if let Some(labels) = labels {
        if let Some(idx) = labels.iter().position(|l| l == label) {
            let cls = idx.min(254) as u8;
            return Some(cls);
        }
    }
    match label.parse::<u8>() {
        Ok(v) if v != UNCLASSIFIED => Some(v),
        _ => None,
    }
}

/// Find connected regions of same-class pixels and return bounding boxes.
/// If `skip` is Some(class), that class is excluded from instance detection
/// (e.g., background in multi-channel argmaxed masks).
#[instrument(skip_all)]
pub fn mask_instance(mask: &[u8], width: usize, skip: Option<u8>) -> Vec<Box2D> {
    let offsets = [-1, 1, -(width as isize), width as isize];
    let mut visited = vec![false; mask.len()];
    let mut boxes = Vec::new();
    for ind in 0..mask.len() {
        if visited[ind] {
            continue;
        }
        let val = mask[ind];
        if skip == Some(val) {
            continue;
        }
        boxes.push(flood_fill(mask, &mut visited, ind, &offsets, width));
    }

    boxes
}

fn flood_fill(
    mask: &[u8],
    visited: &mut [bool],
    ind: usize,
    offsets: &[isize],
    width: usize,
) -> Box2D {
    let mut stack = vec![ind];
    let mut max_x = ind % width;
    let mut min_x = ind % width;
    let mut max_y = ind / width;
    let mut min_y = ind / width;
    while let Some(ind) = stack.pop() {
        max_x = max_x.max(ind % width);
        min_x = min_x.min(ind % width);
        max_y = max_y.max(ind / width);
        min_y = min_y.min(ind / width);
        let mut valid = get_valid_neighbours(mask, visited, ind, offsets, width);
        for ind in &valid {
            visited[*ind] = true;
        }
        stack.append(&mut valid);
    }
    Box2D {
        center_x: (max_x + min_x) as f32 / 2.0 / width as f32,
        center_y: (max_y + min_y) as f32 / 2.0 / (mask.len() / width) as f32,
        width: (max_x - min_x) as f32 / width as f32,
        height: (max_y - min_y) as f32 / (mask.len() / width) as f32,
        label: mask[ind],
    }
}

fn get_valid_neighbours(
    mask: &[u8],
    visited: &mut [bool],
    ind: usize,
    offsets: &[isize],
    width: usize,
) -> Vec<usize> {
    let r = mask[ind];
    let col = ind % width;
    offsets
        .iter()
        .filter_map(|x| {
            let new_ind = ind as isize + *x;
            if new_ind < 0 || new_ind >= mask.len() as isize {
                return None;
            }
            let new_ind = new_ind as usize;
            // Check horizontal neighbors don't wrap rows
            if *x == -1 && col == 0 {
                return None;
            }
            if *x == 1 && col == width - 1 {
                return None;
            }
            if visited[new_ind] {
                return None;
            }
            let r_ = mask[new_ind];
            if r == r_ {
                Some(new_ind)
            } else {
                None
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mask_instance_no_skip_includes_class_zero() {
        // 3x3 mask: class 0 fills the top row, class 1 fills the bottom rows
        #[rustfmt::skip]
        let mask = vec![
            0, 0, 0,
            1, 1, 1,
            1, 1, 1,
        ];
        let boxes = mask_instance(&mask, 3, None);
        let labels: Vec<u8> = boxes.iter().map(|b| b.label).collect();
        assert!(
            labels.contains(&0),
            "class 0 should be included when skip is None"
        );
        assert!(labels.contains(&1), "class 1 should be included");
    }

    #[test]
    fn test_mask_instance_skip_class_zero() {
        #[rustfmt::skip]
        let mask = vec![
            0, 0, 0,
            1, 1, 1,
            1, 1, 1,
        ];
        let boxes = mask_instance(&mask, 3, Some(0));
        let labels: Vec<u8> = boxes.iter().map(|b| b.label).collect();
        assert!(!labels.contains(&0), "class 0 should be skipped");
        assert!(labels.contains(&1), "class 1 should still be included");
    }

    #[test]
    fn test_mask_instance_skip_last_channel() {
        // Simulates argmaxed 3-channel mask where channel 2 is background
        #[rustfmt::skip]
        let mask = vec![
            2, 2, 2,
            0, 0, 2,
            1, 1, 2,
        ];
        let boxes = mask_instance(&mask, 3, Some(2));
        let labels: Vec<u8> = boxes.iter().map(|b| b.label).collect();
        assert!(!labels.contains(&2), "background channel should be skipped");
        assert!(labels.contains(&0), "class 0 should be included");
        assert!(labels.contains(&1), "class 1 should be included");
    }

    #[test]
    fn test_argmax_slice_large_index() {
        // Slice with 256 elements where the max is at index 255
        let mut data = vec![0u8; 256];
        data[255] = 1;
        assert_eq!(
            argmax_slice(&data),
            255,
            "index 255 should be returned as-is"
        );
    }

    #[test]
    fn test_argmax_slice_overflow_clamps() {
        // Slice with 257 elements where the max is at index 256
        // Without clamping this would wrap to 0 via `as u8`
        let mut data = vec![0u8; 257];
        data[256] = 1;
        assert_eq!(
            argmax_slice(&data),
            UNCLASSIFIED,
            "index >= 256 should clamp to UNCLASSIFIED"
        );
    }
}
