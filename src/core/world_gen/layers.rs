//! # Layer implementations
//!
//! This module contains all concrete [`BiomeLayer`](crate::biome_layer::BiomeLayer)
//! implementations that form the biome generation pipeline.
//!
//! ## Module organisation
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`island`] | Root layers — [`LayerIsland`] and [`LayerRiverInit`] |
//! | [`zoom`] | Resolution-doubling layers — [`LayerZoom`], [`LayerFuzzyZoom`], [`LayerVoronoiZoom`] |
//! | [`modifiers`] | Post-processing modifiers — [`LayerAddIsland`], [`LayerEdge`], [`LayerSmooth`] |
//! | [`biome_layer_impl`] | Biome assignment and transition — [`LayerBiome`], [`LayerBiomeEdge`], [`LayerDiffusion`] |

pub mod island;
pub mod zoom;
pub mod modifiers;
pub mod biome_layer_impl;

pub use island::{LayerIsland, LayerRiverInit};
pub use zoom::{LayerZoom, LayerFuzzyZoom, LayerVoronoiZoom};
pub use modifiers::{LayerAddIsland, LayerEdge, LayerSmooth};
pub use biome_layer_impl::{LayerBiome};
