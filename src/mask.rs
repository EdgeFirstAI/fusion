// Copyright 2025 Au-Zone Technologies Inc.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use edgefirst_schemas::{edgefirst_msgs::Mask, serde_cdr};
use itertools::Itertools;
use log::error;
use tokio::sync::Mutex;
use tracing::{info_span, instrument};
use zenoh::Session;

use crate::{args::Args, drain_recv, DrainRecvTimeoutSettings};

// Finds the argmax of the slice. Panics if the slice is empty
pub fn argmax_slice<T: Ord>(slice: &[T]) -> u8 {
    slice.iter().position_max().unwrap() as u8
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Box2D {
    pub center_x: f32,
    pub center_y: f32,
    pub width: f32,
    pub height: f32,
    pub label: u8,
}

pub async fn mask_handler(session: Session, args: Args, mask: Arc<Mutex<Option<Mask>>>) {
    let mask_sub = session
        .declare_subscriber(args.mask_topic.clone())
        .await
        .expect("Failed to declare Zenoh subscriber");
    let mut timeout = DrainRecvTimeoutSettings::default();
    loop {
        let sample = match drain_recv(&mask_sub, &mut timeout).await {
            Some(v) => v,
            None => continue,
        };

        let mut new_mask: Mask = match info_span!("mask_deserialize")
            .in_scope(|| serde_cdr::deserialize::<Mask>(&sample.payload().to_bytes()))
        {
            Ok(v) => v,
            Err(e) => {
                error!("Failed to deserialize message: {e:?}");
                continue;
            }
        };

        if new_mask.encoding == "zstd" {
            new_mask.encoding = "".to_owned();
            new_mask.mask = zstd::decode_all(new_mask.mask.as_slice())
                .expect("Cannot decompress zstd encoded mask");
        }

        if new_mask.width == 0 || new_mask.height == 0 {
            continue;
        }
        let mask_classes = new_mask.mask.len() / new_mask.width as usize / new_mask.height as usize;
        let mask_argmax: Vec<u8> = new_mask
            .mask
            .chunks_exact(mask_classes)
            .map(argmax_slice)
            .collect();
        new_mask.mask = mask_argmax;
        let mut guard = mask.lock().await;
        *guard = Some(new_mask);
    }
}

#[instrument(skip_all)]
pub fn mask_instance(mask: &[u8], width: usize) -> Vec<Box2D> {
    let offsets = [-1, 1, -(width as isize), width as isize];
    let mut visited = vec![false; mask.len()];
    let mut boxes = Vec::new();
    for ind in 0..mask.len() {
        if visited[ind] {
            continue;
        }
        let val = mask[ind];
        if val == 0 {
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
        let mut valid = get_valid_neighbours(mask, visited, ind, offsets);
        for ind in &valid {
            visited[*ind] = true;
        }
        stack.append(&mut valid);
    }
    Box2D {
        center_x: (max_x + min_x) as f32 / 2.0 / width as f32,
        center_y: (max_y + min_y) as f32 / 2.0 / (mask.len() / width) as f32,
        width: (max_x - min_x) as f32 / 2.0 / width as f32,
        height: (max_y - min_y) as f32 / 2.0 / (mask.len() / width) as f32,
        label: mask[ind],
    }
}

fn get_valid_neighbours(
    mask: &[u8],
    visited: &mut [bool],
    ind: usize,
    offsets: &[isize],
) -> Vec<usize> {
    let r = mask[ind];
    if r == 0 {
        return Vec::new();
    }
    offsets
        .iter()
        .filter_map(|x| {
            let new_ind = ind as isize + *x;
            if new_ind < 0 || new_ind >= mask.len() as isize {
                return None;
            }
            let new_ind = new_ind as usize;
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
