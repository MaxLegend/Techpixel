// =============================================================================
// QubePixel — Chunk  (columnar voxel chunk with configurable XYZ dimensions)
//
// Dimensions are read once from config (assets/chunk_config.json).
// Default: 16 × 256 × 16 (columnar, one chunk per XZ column).
//
// Block storage: flat Vec<u8>, index = x*SY*SZ + y*SZ + z
// =============================================================================

use crate::{ flow_debug_log};
use crate::core::config;
use crate::core::gameobjects::block::{BlockRegistry, BlockShape, remap_face_for_rotation};
use crate::screens::game_3d_pipeline::{Vertex3D, snorm8, unorm8};
use crate::core::gameobjects::texture_atlas::TextureAtlasLayout;

// ---------------------------------------------------------------------------
// Dimension helpers — read from global config (initialised from JSON at startup)
// ---------------------------------------------------------------------------

#[inline(always)] fn csx() -> usize { config::chunk_size_x() }
#[inline(always)] fn csy() -> usize { config::chunk_size_y() }
#[inline(always)] fn csz() -> usize { config::chunk_size_z() }

/// Flat index into the block array: [x * SY * SZ + y * SZ + z]
#[inline(always)]
fn flat_idx(x: usize, y: usize, z: usize) -> usize {
    x * csy() * csz() + y * csz() + z
}

// ---------------------------------------------------------------------------
// Chunk
// ---------------------------------------------------------------------------

pub struct Chunk {
    pub cx: i32,
    pub cy: i32,
    pub cz: i32,
    /// Flat block array: index = x*SY*SZ + y*SZ + z. 0 = Air.
    pub(crate) blocks: Vec<u8>,
    /// Per-block rotation / facing direction (same indexing as `blocks`).
    /// 0 = default orientation. See `block::ROT_FACE_MAP` for encoding.
    pub(crate) rotations: Vec<u8>,
    /// Discrete fluid level per block (0–8). Only meaningful when block is a fluid.
    /// 0 = empty, 1–7 = flowing, 8 = source. Same indexing as `blocks`.
    pub(crate) fluid_levels: Vec<u8>,
    pub mesh_dirty: bool,
    /// Per-XZ-column biome foliage tint (RGB multiplier). Index = bx * CSZ + bz.
    /// Applied to blocks with `biome_tint: true` (grass, leaves, etc.).
    pub biome_foliage_colors: Vec<[f32; 3]>,
    /// Per-XZ-column biome water tint (RGB multiplier). Index = bx * CSZ + bz.
    /// Applied to fluid block vertex colours.
    pub biome_water_colors: Vec<[f32; 3]>,
    /// Averaged biome ambient tint for the whole chunk column.
    /// Used to tint the global ambient light when the camera is in this chunk.
    pub biome_ambient_tint: [f32; 3],
}

impl Chunk {
    pub fn new(cx: i32, cy: i32, cz: i32) -> Self {
        let vol = csx() * csy() * csz();
        let cols = csx() * csz();
        Self {
            cx,
            cy,
            cz,
            blocks: vec![0u8; vol],
            rotations: vec![0u8; vol],
            fluid_levels: vec![0u8; vol],
            mesh_dirty: true,
            biome_foliage_colors: vec![[1.0f32; 3]; cols],
            biome_water_colors:   vec![[1.0f32; 3]; cols],
            biome_ambient_tint:   [1.0f32; 3],
        }
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize, z: usize) -> u8 {
        self.blocks[flat_idx(x, y, z)]
    }

    #[allow(dead_code)]
    pub fn set(&mut self, x: usize, y: usize, z: usize, block_id: u8) {
        self.blocks[flat_idx(x, y, z)] = block_id;
        self.mesh_dirty = true;
    }

    /// Write a block without marking the chunk dirty (used during terrain generation).
    pub(crate) fn set_gen(&mut self, x: usize, y: usize, z: usize, block_id: u8) {
        self.blocks[flat_idx(x, y, z)] = block_id;
    }

    /// Write a fluid block with discrete level during terrain generation.
    /// Use `crate::core::fluid::FLUID_SOURCE_LEVEL` (= 8) for source blocks.
    pub(crate) fn set_gen_fluid(&mut self, x: usize, y: usize, z: usize, block_id: u8, level: u8) {
        let idx = flat_idx(x, y, z);
        self.blocks[idx] = block_id;
        self.fluid_levels[idx] = level;
    }

    /// Legacy alias — kept for backward compatibility.
    pub(crate) fn set_gen_water(&mut self, x: usize, y: usize, z: usize, block_id: u8, level: u8) {
        self.set_gen_fluid(x, y, z, block_id, level);
    }

    /// Write a block with explicit rotation during terrain generation (no dirty flag).
    pub(crate) fn set_gen_with_rotation(&mut self, x: usize, y: usize, z: usize, block_id: u8, rotation: u8) {
        let idx = flat_idx(x, y, z);
        self.blocks[idx] = block_id;
        self.rotations[idx] = rotation;
    }

    /// Set a block with rotation and mark the chunk dirty (player placement).
    pub fn set_with_rotation(&mut self, x: usize, y: usize, z: usize, block_id: u8, rotation: u8) {
        let idx = flat_idx(x, y, z);
        self.blocks[idx] = block_id;
        self.rotations[idx] = rotation;
        self.mesh_dirty = true;
    }

    /// Get the rotation value stored at a local block position.
    #[inline]
    pub fn get_rotation(&self, x: usize, y: usize, z: usize) -> u8 {
        self.rotations[flat_idx(x, y, z)]
    }

    /// Get discrete fluid level (0–8) at local position.
    #[inline]
    pub fn get_fluid_level(&self, x: usize, y: usize, z: usize) -> u8 {
        self.fluid_levels[flat_idx(x, y, z)]
    }

    /// Get fluid level as a normalized float (0.0–1.0) for rendering.
    #[inline]
    pub fn get_fluid_level_f32(&self, x: usize, y: usize, z: usize) -> f32 {
        self.fluid_levels[flat_idx(x, y, z)] as f32 / 8.0
    }

    /// Legacy alias — kept for backward compatibility.
    #[inline]
    pub fn get_water_level(&self, x: usize, y: usize, z: usize) -> u8 {
        self.get_fluid_level(x, y, z)
    }

    pub fn is_all_air(&self) -> bool {
        self.blocks.iter().all(|&b| b == 0)
    }

    // -----------------------------------------------------------------------
    // Fluid mesh builder (variable-height transparent fluid surfaces)
    // -----------------------------------------------------------------------

    /// Build a separate mesh for fluid blocks with variable surface height.
    ///
    /// Fluid faces are only emitted where fluid borders non-fluid (air or solid).
    /// The top face height is determined by `fluid_levels`.
    /// Returns `(vertices, indices)` using the same `Vertex3D` format.
    /// Alpha is encoded in `color_ao[3]` for the fluid shader.
    ///
    /// `fluid_ids` — slice of all registered fluid block IDs.
    pub fn build_fluid_mesh<F>(
        &self,
        registry: &BlockRegistry,
        atlas: &TextureAtlasLayout,
        fluid_ids: &[u8],
        get_neighbor: F,
    ) -> (Vec<Vertex3D>, Vec<u32>)
    where
        F: Fn(i32, i32, i32) -> (u8, u8),
    {
        let sx = csx();
        let sy = csy();
        let sz = csz();

        let wx_base = self.cx * sx as i32;
        let wy_base = self.cy * sy as i32;
        let wz_base = self.cz * sz as i32;

        let mut vertices: Vec<Vertex3D> = Vec::with_capacity(2048);
        let mut indices:  Vec<u32>      = Vec::with_capacity(3072);

        // Maximum visual height for a source/full fluid block (level 8 maps to 0.875).
        // Using 7/8 = 0.875 gives a visible surface gap even for sources.
        const MAX_FLUID_VISUAL_LEVEL: f32 = 0.875;

        // Unified neighbor lookup → (block_id, discrete_level).
        let get_nb = |nwx: i32, nwy: i32, nwz: i32| -> (u8, u8) {
            let lx = nwx - wx_base;
            let ly = nwy - wy_base;
            let lz = nwz - wz_base;
            if lx >= 0 && lx < sx as i32
                && ly >= 0 && ly < sy as i32
                && lz >= 0 && lz < sz as i32
            {
                let idx = flat_idx(lx as usize, ly as usize, lz as usize);
                (self.blocks[idx], self.fluid_levels[idx])
            } else {
                get_neighbor(nwx, nwy, nwz)
            }
        };

        for bx in 0..sx {
            for by in 0..sy {
                for bz in 0..sz {
                    let block_id = self.blocks[flat_idx(bx, by, bz)];

                    // Skip non-fluid blocks
                    if !fluid_ids.contains(&block_id) { continue; }

                    let level_u8 = self.fluid_levels[flat_idx(bx, by, bz)];
                    if level_u8 == 0 { continue; }

                    // Normalized level [0..1] for visual height calculations
                    let level = (level_u8 as f32 / 8.0).min(MAX_FLUID_VISUAL_LEVEL);

                    let def = match registry.get(block_id) {
                        Some(d) => d,
                        None => continue,
                    };

                    let ox = wx_base as f32 + bx as f32;
                    let oy = wy_base as f32 + by as f32;
                    let oz = wz_base as f32 + bz as f32;

                    let wx = wx_base + bx as i32;
                    let wy = wy_base + by as i32;
                    let wz = wz_base + bz as i32;

                    // Apply biome water tint to the base colour
                    let base_c = def.color_for_face(2); // use top color for all faces
                    let water_tint = self.biome_water_colors[bx * csz() + bz];
                    let c = [
                        base_c[0] * water_tint[0],
                        base_c[1] * water_tint[1],
                        base_c[2] * water_tint[2],
                    ];
                    let uv = atlas.uv_for(def.texture_for_face(2).unwrap_or(""));
                    let mat = &def.material;

                    // Depth-dependent alpha: deeper fluid is more opaque
                    let alpha = (100.0 + level * 120.0).min(230.0) as u8;

                    // Emission for luminous fluids (e.g. lava)
                    let emi = &def.emission;
                    let emission_data = if emi.emit_light {
                        [emi.light_color[0], emi.light_color[1], emi.light_color[2], -emi.light_intensity.abs()]
                    } else {
                        [0.0f32; 4]
                    };

                    let is_same_fluid = |bid: u8| -> bool { bid == block_id };
                    let is_any_fluid  = |bid: u8| -> bool { fluid_ids.contains(&bid) };

                    // --- Face neighbours (for culling) ---
                    let npx = get_nb(wx + 1, wy, wz).0;
                    let nnx = get_nb(wx - 1, wy, wz).0;
                    let npy = get_nb(wx, wy + 1, wz).0;
                    let nny = get_nb(wx, wy - 1, wz).0;
                    let npz = get_nb(wx, wy, wz + 1).0;
                    let nnz = get_nb(wx, wy, wz - 1).0;

                    // --- Smooth per-corner surface heights ---
                    //
                    // For each of the 4 top-face corners, average the effective visual
                    // level of the 4 blocks that share that corner (current + 3 neighbours).
                    // A block is "submerged" (same fluid directly above it) → treat as
                    // MAX_FLUID_VISUAL_LEVEL so the surface connects to the column above.
                    // Air neighbours contribute 0 and are skipped in the average.

                    let eff_level = |dx: i32, dz: i32| -> f32 {
                        if dx == 0 && dz == 0 {
                            if npy == block_id { MAX_FLUID_VISUAL_LEVEL }
                            else { level }
                        } else {
                            let (bid, bl) = get_nb(wx + dx, wy, wz + dz);
                            if bid != block_id { return 0.0; }
                            let above = get_nb(wx + dx, wy + 1, wz + dz).0;
                            if above == block_id { MAX_FLUID_VISUAL_LEVEL }
                            else { (bl as f32 / 8.0).min(MAX_FLUID_VISUAL_LEVEL) }
                        }
                    };

                    let corner_h = |dx: i32, dz: i32| -> f32 {
                        let levels = [
                            eff_level( 0,  0),
                            eff_level(dx,  0),
                            eff_level( 0, dz),
                            eff_level(dx, dz),
                        ];
                        let mut total = 0.0f32;
                        let mut count = 0u32;
                        for l in &levels {
                            if *l > 0.0 { total += l; count += 1; }
                        }
                        if count > 0 { (total / count as f32).max(0.0625) } else { 0.0625 }
                    };

                    let h_mm = corner_h(-1, -1); // (-x, -z) corner
                    let h_pm = corner_h( 1, -1); // (+x, -z) corner
                    let h_pp = corner_h( 1,  1); // (+x, +z) corner
                    let h_mp = corner_h(-1,  1); // (-x, +z) corner

                    // +X face: show if neighbor is air or a different fluid
                    if npx == 0 || (is_any_fluid(npx) && !is_same_fluid(npx)) {
                        emit_water_face(&mut vertices, &mut indices,
                            ox, oy, oz, 0, h_mm, h_pm, h_pp, h_mp,
                            c, alpha, uv, [mat.roughness, mat.metalness], emission_data);
                    }
                    // -X face
                    if nnx == 0 || (is_any_fluid(nnx) && !is_same_fluid(nnx)) {
                        emit_water_face(&mut vertices, &mut indices,
                            ox, oy, oz, 1, h_mm, h_pm, h_pp, h_mp,
                            c, alpha, uv, [mat.roughness, mat.metalness], emission_data);
                    }
                    // +Y face (top surface): show if block above is NOT the same fluid
                    if !is_same_fluid(npy) {
                        emit_water_face(&mut vertices, &mut indices,
                            ox, oy, oz, 2, h_mm, h_pm, h_pp, h_mp,
                            c, alpha, uv, [mat.roughness, mat.metalness], emission_data);
                    }
                    // -Y face (bottom)
                    if nny == 0 || (is_any_fluid(nny) && !is_same_fluid(nny)) {
                        emit_water_face(&mut vertices, &mut indices,
                            ox, oy, oz, 3, h_mm, h_pm, h_pp, h_mp,
                            c, alpha, uv, [mat.roughness, mat.metalness], emission_data);
                    }
                    // +Z face
                    if npz == 0 || (is_any_fluid(npz) && !is_same_fluid(npz)) {
                        emit_water_face(&mut vertices, &mut indices,
                            ox, oy, oz, 4, h_mm, h_pm, h_pp, h_mp,
                            c, alpha, uv, [mat.roughness, mat.metalness], emission_data);
                    }
                    // -Z face
                    if nnz == 0 || (is_any_fluid(nnz) && !is_same_fluid(nnz)) {
                        emit_water_face(&mut vertices, &mut indices,
                            ox, oy, oz, 5, h_mm, h_pm, h_pp, h_mp,
                            c, alpha, uv, [mat.roughness, mat.metalness], emission_data);
                    }
                }
            }
        }

        (vertices, indices)
    }

    /// Legacy alias — builds a mesh for a single fluid type.
    pub fn build_water_mesh<F>(
        &self,
        registry: &BlockRegistry,
        atlas: &TextureAtlasLayout,
        water_block_id: u8,
        get_neighbor: F,
    ) -> (Vec<Vertex3D>, Vec<u32>)
    where
        F: Fn(i32, i32, i32) -> (u8, u8),
    {
        self.build_fluid_mesh(registry, atlas, &[water_block_id], get_neighbor)
    }

    // -----------------------------------------------------------------------
    // Transparent (glass) mesh builder — full-block alpha-blended faces
    // -----------------------------------------------------------------------

    /// Build a mesh containing only the faces of *transparent solid* blocks
    /// (e.g. glass). One full-cube face is emitted per visible face. Face culling:
    ///
    /// - Glass face vs air / fluid / model:  EMIT (face is visible)
    /// - Glass face vs same block id:        CULL (internal seam invisible)
    /// - Glass face vs opaque solid:         CULL (hidden behind opaque)
    /// - Glass face vs different glass id:   EMIT (different tints interact)
    ///
    /// Texture alpha multiplies the per-fragment opacity, so transparent texels
    /// (e.g. the centre of a frosted-glass tile) let light through naturally
    /// during alpha-blending. The vertex `color_ao[3]` is set from the block's
    /// configured `glass.opacity` so the fragment shader has both texture alpha
    /// and per-block opacity to combine.
    pub fn build_transparent_mesh<F>(
        &self,
        registry: &BlockRegistry,
        atlas: &TextureAtlasLayout,
        get_neighbor: F,
    ) -> (Vec<Vertex3D>, Vec<u32>)
    where
        F: Fn(i32, i32, i32) -> u8,
    {
        let sx = csx();
        let sy = csy();
        let sz = csz();

        let wx_base = self.cx * sx as i32;
        let wy_base = self.cy * sy as i32;
        let wz_base = self.cz * sz as i32;

        let mut vertices: Vec<Vertex3D> = Vec::with_capacity(256);
        let mut indices:  Vec<u32>      = Vec::with_capacity(384);

        // Read-through neighbour fetch (in-chunk → blocks, cross-chunk → get_neighbor).
        let get_nb = |nwx: i32, nwy: i32, nwz: i32| -> u8 {
            let lx = nwx - wx_base;
            let ly = nwy - wy_base;
            let lz = nwz - wz_base;
            if lx >= 0 && lx < sx as i32
                && ly >= 0 && ly < sy as i32
                && lz >= 0 && lz < sz as i32 {
                self.blocks[flat_idx(lx as usize, ly as usize, lz as usize)]
            } else {
                get_neighbor(nwx, nwy, nwz)
            }
        };

        // A neighbour hides this face if it is the same block id (merge identical
        // glass) OR a fully-opaque solid cube. Other alpha/cutout neighbours stay
        // visible so you can see through to them.
        let face_hidden_by = |neighbour_id: u8, self_id: u8| -> bool {
            if neighbour_id == self_id { return true; }
            match registry.get(neighbour_id) {
                Some(d) => d.solid
                    && !d.renders_in_alpha_pass()
                    && d.model.is_none()
                    && matches!(d.block_shape, BlockShape::Cube),
                None => false,
            }
        };

        for bx in 0..sx {
            for by in 0..sy {
                for bz in 0..sz {
                    let block_id = self.blocks[flat_idx(bx, by, bz)];
                    if block_id == 0 { continue; }
                    let def = match registry.get(block_id) {
                        Some(d) => d,
                        None => continue,
                    };
                    if !def.renders_in_alpha_pass() { continue; }

                    let ox = wx_base as f32 + bx as f32;
                    let oy = wy_base as f32 + by as f32;
                    let oz = wz_base as f32 + bz as f32;
                    let wx = wx_base + bx as i32;
                    let wy = wy_base + by as i32;
                    let wz = wz_base + bz as i32;

                    let mat = &def.material;
                    let material_data = [mat.roughness, mat.metalness];
                    let emi = &def.emission;
                    let emission_data = if emi.emit_light {
                        [emi.light_color[0], emi.light_color[1], emi.light_color[2], -emi.light_intensity.abs()]
                    } else {
                        [0.0; 4]
                    };

                    // Vertex alpha feeds the shader as `out_alpha = texture.a × vertex_alpha`.
                    // • Texture-transparent blocks (holes / per-texel glass) pass 255 so the
                    //   opacity is taken entirely from the texture (opaque texels stay solid,
                    //   clear texels become holes).
                    // • Legacy uniform-opacity glass (no alpha in its texture) keeps its
                    //   configured `glass.opacity`, clamped so it never fully vanishes.
                    let alpha = if def.tex_has_alpha {
                        255u8
                    } else {
                        (def.glass.opacity.clamp(0.0, 1.0).max(0.1) * 255.0) as u8
                    };

                    let rot = self.get_rotation(bx, by, bz);
                    let tint_face = |world_face: u8| -> [f32; 3] {
                        def.color_for_face(remap_face_for_rotation(rot, world_face))
                    };
                    let tex_face = |world_face: u8| -> Option<&str> {
                        def.texture_for_face(remap_face_for_rotation(rot, world_face))
                    };

                    let nb = [
                        get_nb(wx + 1, wy, wz), // +X
                        get_nb(wx - 1, wy, wz), // -X
                        get_nb(wx, wy + 1, wz), // +Y
                        get_nb(wx, wy - 1, wz), // -Y
                        get_nb(wx, wy, wz + 1), // +Z
                        get_nb(wx, wy, wz - 1), // -Z
                    ];

                    for face in 0..6u8 {
                        if face_hidden_by(nb[face as usize], block_id) { continue; }
                        let c  = tint_face(face);
                        let uv = atlas.uv_for(tex_face(face).unwrap_or(""));
                        emit_face_lod_with_alpha(
                            &mut vertices, &mut indices,
                            ox, oy, oz, face, c, alpha, uv, 1.0,
                            material_data, emission_data,
                        );
                    }
                }
            }
        }

        (vertices, indices)
    }

    // -----------------------------------------------------------------------
    // Foliage mesh builder — Cross and CropGrid shapes (double-sided quads)
    // -----------------------------------------------------------------------

    /// Build a mesh for all foliage blocks (Cross / CropGrid shapes) in this chunk.
    ///
    /// Cross    — two diagonal quads, each rendered double-sided.
    /// CropGrid — four axis-aligned quads in a # pattern, each double-sided.
    ///
    /// Returns `(vertices, indices)` in the same `Vertex3D` format as other builders.
    /// This mesh is rendered without backface culling (foliage is always two-sided).
    pub fn build_foliage_mesh(
        &self,
        registry: &BlockRegistry,
        atlas: &TextureAtlasLayout,
    ) -> (Vec<Vertex3D>, Vec<u32>) {
        let sx = csx();
        let sy = csy();
        let sz = csz();

        let wx_base = self.cx * sx as i32;
        let wy_base = self.cy * sy as i32;
        let wz_base = self.cz * sz as i32;

        let mut vertices: Vec<Vertex3D> = Vec::new();
        let mut indices:  Vec<u32>      = Vec::new();

        for bx in 0..sx {
            for by in 0..sy {
                for bz in 0..sz {
                    let block_id = self.blocks[flat_idx(bx, by, bz)];
                    if block_id == 0 { continue; }
                    let def = match registry.get(block_id) {
                        Some(d) => d,
                        None => continue,
                    };
                    let is_cross     = def.block_shape == BlockShape::Cross;
                    let is_cropgrid  = def.block_shape == BlockShape::CropGrid;
                    if !is_cross && !is_cropgrid { continue; }

                    let ox = wx_base as f32 + bx as f32;
                    let oy = wy_base as f32 + by as f32;
                    let oz = wz_base as f32 + bz as f32;

                    let col_idx = bx * csz() + bz;
                    let btint = if def.biome_tint {
                        self.biome_foliage_colors[col_idx]
                    } else {
                        [1.0f32; 3]
                    };
                    let base_c = def.color_for_face(2); // use top face colour for all quads
                    let color = [
                        base_c[0] * btint[0],
                        base_c[1] * btint[1],
                        base_c[2] * btint[2],
                    ];
                    let uv = atlas.uv_for(def.texture_for_face(2).unwrap_or(""));
                    let mat = &def.material;
                    let material_data = [mat.roughness, mat.metalness];
                    let emi = &def.emission;
                    let emission_data = if emi.emit_light {
                        [emi.light_color[0], emi.light_color[1], emi.light_color[2], -emi.light_intensity.abs()]
                    } else {
                        [0.0f32; 4]
                    };

                    if is_cross {
                        // Quad A: NW→SE diagonal
                        emit_foliage_quad(
                            &mut vertices, &mut indices,
                            [
                                [ox+0.1464, oy,   oz+0.1464],
                                [ox+0.8536, oy,   oz+0.8536],
                                [ox+0.8536, oy+1.0, oz+0.8536],
                                [ox+0.1464, oy+1.0, oz+0.1464],
                            ],
                            color, uv, material_data, emission_data,
                        );
                        // Quad B: NE→SW diagonal
                        emit_foliage_quad(
                            &mut vertices, &mut indices,
                            [
                                [ox+0.8536, oy,   oz+0.1464],
                                [ox+0.1464, oy,   oz+0.8536],
                                [ox+0.1464, oy+1.0, oz+0.8536],
                                [ox+0.8536, oy+1.0, oz+0.1464],
                            ],
                            color, uv, material_data, emission_data,
                        );
                    } else {
                        // CropGrid — four axis-aligned quads
                        // X1: z = oz+0.333
                        emit_foliage_quad(
                            &mut vertices, &mut indices,
                            [
                                [ox,     oy,     oz+0.333],
                                [ox+1.0, oy,     oz+0.333],
                                [ox+1.0, oy+1.0, oz+0.333],
                                [ox,     oy+1.0, oz+0.333],
                            ],
                            color, uv, material_data, emission_data,
                        );
                        // X2: z = oz+0.667
                        emit_foliage_quad(
                            &mut vertices, &mut indices,
                            [
                                [ox,     oy,     oz+0.667],
                                [ox+1.0, oy,     oz+0.667],
                                [ox+1.0, oy+1.0, oz+0.667],
                                [ox,     oy+1.0, oz+0.667],
                            ],
                            color, uv, material_data, emission_data,
                        );
                        // Z1: x = ox+0.333
                        emit_foliage_quad(
                            &mut vertices, &mut indices,
                            [
                                [ox+0.333, oy,     oz    ],
                                [ox+0.333, oy,     oz+1.0],
                                [ox+0.333, oy+1.0, oz+1.0],
                                [ox+0.333, oy+1.0, oz    ],
                            ],
                            color, uv, material_data, emission_data,
                        );
                        // Z2: x = ox+0.667
                        emit_foliage_quad(
                            &mut vertices, &mut indices,
                            [
                                [ox+0.667, oy,     oz    ],
                                [ox+0.667, oy,     oz+1.0],
                                [ox+0.667, oy+1.0, oz+1.0],
                                [ox+0.667, oy+1.0, oz    ],
                            ],
                            color, uv, material_data, emission_data,
                        );
                    }
                }
            }
        }

        flow_debug_log!(
            "Chunk", "build_foliage_mesh",
            "Chunk ({},{},{}) => {} foliage verts, {} indices",
            self.cx, self.cy, self.cz, vertices.len(), indices.len()
        );

        (vertices, indices)
    }

    // -----------------------------------------------------------------------
    // Full-detail mesh (LOD 0)
    // -----------------------------------------------------------------------

    pub fn build_mesh<F>(
        &self,
        registry: &BlockRegistry,
        atlas: &TextureAtlasLayout,
        get_neighbor: F,
    ) -> (Vec<Vertex3D>, Vec<u32>)
    where
        F: Fn(i32, i32, i32) -> u8,
    {
        self.build_mesh_lod(registry, atlas, get_neighbor, 0, 0)
    }

    // -----------------------------------------------------------------------
    // LOD mesh builder
    // -----------------------------------------------------------------------

    /// Build a mesh with configurable LOD level.
    ///
    /// * `lod_level = 0` → step = 1 (full detail)
    /// * `lod_level = 1` → step = 2 (2× coarser)
    /// * `lod_level = 2` → step = 4 (4× coarser)
    ///
    /// Each "super-block" uses the dominant non-air block in its step³ region.
    pub fn build_mesh_lod<F>(
        &self,
        registry: &BlockRegistry,
        atlas: &TextureAtlasLayout,
        get_neighbor: F,
        lod_level: u8,
        water_block_id: u8,
    ) -> (Vec<Vertex3D>, Vec<u32>)
    where
        F: Fn(i32, i32, i32) -> u8,
    {
        if self.is_all_air() {
            return (Vec::new(), Vec::new());
        }

        let sx = csx();
        let sy = csy();
        let sz = csz();

        let step = 1usize << (lod_level.min(2) as usize); // 1, 2, or 4
        let cells_x = sx / step;
        let cells_y = sy / step;
        let cells_z = sz / step;
        let scale   = step as f32;

        let wx_base = self.cx * sx as i32;
        let wy_base = self.cy * sy as i32;
        let wz_base = self.cz * sz as i32;

        let cap = if lod_level == 0 { 4096 } else { 512 };
        let mut vertices: Vec<Vertex3D> = Vec::with_capacity(cap);
        let mut indices:  Vec<u32>      = Vec::with_capacity(cap * 3 / 2);

        for cx_cell in 0..cells_x {
            for cy_cell in 0..cells_y {
                for cz_cell in 0..cells_z {
                    let bx0 = cx_cell * step;
                    let by0 = cy_cell * step;
                    let bz0 = cz_cell * step;

                    let block_id = self.dominant_block(bx0, by0, bz0, step, sx, sy, sz);
                    if block_id == 0 { continue; }

                    let def = match registry.get(block_id) {
                        Some(d) => d,
                        None => continue,
                    };
                    if !def.solid { continue; }
                    // Skip blocks with custom 3D models — they are rendered as entities
                    if def.model.is_some() { continue; }
                    // Skip alpha-pass blocks (glass / texture-transparent, auto-detected)
                    // — they are meshed via build_transparent_mesh.
                    if def.renders_in_alpha_pass() { continue; }
                    // Skip foliage (Cross/CropGrid) — meshed via build_foliage_mesh.
                    // Skip Slab — meshed via build_slab_mesh.
                    if matches!(def.block_shape, BlockShape::Cross | BlockShape::CropGrid | BlockShape::Slab) { continue; }

                    let ox = wx_base as f32 + bx0 as f32;
                    let oy = wy_base as f32 + by0 as f32;
                    let oz = wz_base as f32 + bz0 as f32;

                    let wx = wx_base + bx0 as i32;
                    let wy = wy_base + by0 as i32;
                    let wz = wz_base + bz0 as i32;

                    let mat = &def.material;
                    let material_data = [mat.roughness, mat.metalness];
                    let emi = &def.emission;
                    let emission_data = if emi.emit_light {
                        // Negative intensity encodes no_self_gi: the fragment shader
                        // skips GI sampling for this block so its own injected emission
                        // doesn't wash over its surface as a yellow/colored overlay.
                        // All light-emitting blocks are self-sufficient light sources
                        // and should never receive GI from their own probes.
                        [emi.light_color[0], emi.light_color[1], emi.light_color[2], -emi.light_intensity.abs()]
                    } else {
                        [0.0, 0.0, 0.0, 0.0]
                    };

                    // Biome foliage tint — applied only to blocks flagged with `biome_tint: true`.
                    let col_idx = bx0 * csz() + bz0;
                    let btint = if def.biome_tint {
                        self.biome_foliage_colors[col_idx]
                    } else {
                        [1.0f32; 3]
                    };

                    // Block rotation: remap world face → block-local face for texture/colour.
                    let rot = self.get_rotation(bx0, by0, bz0);
                    let tint_face = |world_face: u8| -> [f32; 3] {
                        let local_face = remap_face_for_rotation(rot, world_face);
                        let fc = def.color_for_face(local_face);
                        [fc[0] * btint[0], fc[1] * btint[1], fc[2] * btint[2]]
                    };
                    let tex_face = |world_face: u8| -> Option<&str> {
                        def.texture_for_face(remap_face_for_rotation(rot, world_face))
                    };

                    let ws = step as i32;

                    // +X face
                    if self.is_adjacent_air(bx0 + step, by0, bz0, step, 0, wx, wy, wz, sx, sy, sz, &get_neighbor, water_block_id, registry) {
                        let c  = tint_face(0);
                        let uv = atlas.uv_for(tex_face(0).unwrap_or(""));
                        emit_face_lod(&mut vertices, &mut indices, ox, oy, oz, 0, c, uv, scale, material_data, emission_data);
                    }
                    // -X face
                    if self.is_adjacent_air_neg(bx0, by0, bz0, step, 1, wx, wy, wz, sx, sy, sz, &get_neighbor, water_block_id, registry) {
                        let c  = tint_face(1);
                        let uv = atlas.uv_for(tex_face(1).unwrap_or(""));
                        emit_face_lod(&mut vertices, &mut indices, ox, oy, oz, 1, c, uv, scale, material_data, emission_data);
                    }
                    // +Y face
                    if self.is_adjacent_air(bx0, by0 + step, bz0, step, 2, wx, wy, wz, sx, sy, sz, &get_neighbor, water_block_id, registry) {
                        let c  = tint_face(2);
                        let uv = atlas.uv_for(tex_face(2).unwrap_or(""));
                        emit_face_lod(&mut vertices, &mut indices, ox, oy, oz, 2, c, uv, scale, material_data, emission_data);
                    }
                    // -Y face
                    if self.is_adjacent_air_neg(bx0, by0, bz0, step, 3, wx, wy, wz, sx, sy, sz, &get_neighbor, water_block_id, registry) {
                        let c  = tint_face(3);
                        let uv = atlas.uv_for(tex_face(3).unwrap_or(""));
                        emit_face_lod(&mut vertices, &mut indices, ox, oy, oz, 3, c, uv, scale, material_data, emission_data);
                    }
                    // +Z face
                    if self.is_adjacent_air(bx0, by0, bz0 + step, step, 4, wx, wy, wz, sx, sy, sz, &get_neighbor, water_block_id, registry) {
                        let c  = tint_face(4);
                        let uv = atlas.uv_for(tex_face(4).unwrap_or(""));
                        emit_face_lod(&mut vertices, &mut indices, ox, oy, oz, 4, c, uv, scale, material_data, emission_data);
                    }
                    // -Z face
                    if self.is_adjacent_air_neg(bx0, by0, bz0, step, 5, wx, wy, wz, sx, sy, sz, &get_neighbor, water_block_id, registry) {
                        let c  = tint_face(5);
                        let uv = atlas.uv_for(tex_face(5).unwrap_or(""));
                        emit_face_lod(&mut vertices, &mut indices, ox, oy, oz, 5, c, uv, scale, material_data, emission_data);
                    }

                    let _ = ws; // suppress unused warning
                }
            }
        }

        flow_debug_log!(
            "Chunk", "build_mesh_lod",
            "Chunk ({},{},{}) lod={} step={} => {} verts, {} indices",
            self.cx, self.cy, self.cz, lod_level, step,
            vertices.len(), indices.len()
        );

        (vertices, indices)
    }

    // -----------------------------------------------------------------------
    // LOD helpers
    // -----------------------------------------------------------------------

    fn dominant_block(
        &self,
        bx0: usize, by0: usize, bz0: usize,
        step: usize,
        sx: usize, sy: usize, sz: usize,
    ) -> u8 {
        if step == 1 {
            return self.blocks[flat_idx(bx0, by0, bz0)];
        }

        let mid = step / 2;
        let cx = (bx0 + mid).min(sx - 1);
        let cy = (by0 + mid).min(sy - 1);
        let cz = (bz0 + mid).min(sz - 1);
        let center = self.blocks[flat_idx(cx, cy, cz)];
        if center != 0 { return center; }

        for dx in 0..step {
            for dy in 0..step {
                for dz in 0..step {
                    let x = bx0 + dx;
                    let y = by0 + dy;
                    let z = bz0 + dz;
                    if x < sx && y < sy && z < sz {
                        let id = self.blocks[flat_idx(x, y, z)];
                        if id != 0 { return id; }
                    }
                }
            }
        }
        0
    }

    /// Returns true if a block ID should be treated as transparent for face culling.
    /// Air (0), fluid blocks, blocks with custom 3D models, and glass-like solid
    /// transparent blocks all let neighbouring faces show through (so neighbour
    /// faces are NOT culled).
    #[inline]
    fn is_transparent(b: u8, water_block_id: u8, registry: &BlockRegistry) -> bool {
        if b == 0 { return true; }
        if water_block_id != 0 && b == water_block_id { return true; }
        registry.get(b).map(|d| {
            d.model.is_some()
                || d.transparent
                || d.tex_has_alpha   // auto-detected texture transparency (holes / glass)
                || matches!(d.block_shape, BlockShape::Cross | BlockShape::CropGrid)
        }).unwrap_or(false)
    }

    /// Check if the positive-direction adjacent super-block is transparent (face culling).
    #[allow(clippy::too_many_arguments)]
    fn is_adjacent_air<F>(
        &self,
        bx: usize, by: usize, bz: usize,
        step: usize, dir: u8,
        wx: i32, wy: i32, wz: i32,
        sx: usize, sy: usize, sz: usize,
        get_neighbor: &F,
        water_block_id: u8,
        registry: &BlockRegistry,
    ) -> bool
    where F: Fn(i32, i32, i32) -> u8
    {
        let (check_local, world_offset) = match dir {
            0 => (bx < sx, (step as i32, 0, 0)),
            2 => (by < sy, (0, step as i32, 0)),
            4 => (bz < sz, (0, 0, step as i32)),
            _ => return true,
        };

        if check_local {
            let (lx, ly, lz) = match dir {
                0 => (bx,              by + step / 2, bz + step / 2),
                2 => (bx + step / 2,  by,            bz + step / 2),
                4 => (bx + step / 2,  by + step / 2, bz           ),
                _ => (bx, by, bz),
            };
            let lx = lx.min(sx - 1);
            let ly = ly.min(sy - 1);
            let lz = lz.min(sz - 1);
            Self::is_transparent(self.blocks[flat_idx(lx, ly, lz)], water_block_id, registry)
        } else {
            let b = get_neighbor(
                wx + world_offset.0,
                wy + world_offset.1,
                wz + world_offset.2,
            );
            Self::is_transparent(b, water_block_id, registry)
        }
    }

    /// Check negative-direction adjacency.
    #[allow(clippy::too_many_arguments)]
    fn is_adjacent_air_neg<F>(
        &self,
        bx0: usize, by0: usize, bz0: usize,
        step: usize, dir: u8,
        wx: i32, wy: i32, wz: i32,
        sx: usize, sy: usize, sz: usize,
        get_neighbor: &F,
        water_block_id: u8,
        registry: &BlockRegistry,
    ) -> bool
    where F: Fn(i32, i32, i32) -> u8
    {
        let (can_check_local, world_offset) = match dir {
            1 => (bx0 >= step, (-1i32, 0, 0)),
            3 => (by0 >= step, (0, -1i32, 0)),
            5 => (bz0 >= step, (0, 0, -1i32)),
            _ => return true,
        };

        if can_check_local {
            let (lx, ly, lz) = match dir {
                1 => (bx0 - 1,          by0 + step / 2, bz0 + step / 2),
                3 => (bx0 + step / 2,   by0 - 1,        bz0 + step / 2),
                5 => (bx0 + step / 2,   by0 + step / 2, bz0 - 1       ),
                _ => (bx0, by0, bz0),
            };
            let lx = lx.min(sx - 1);
            let ly = ly.min(sy - 1);
            let lz = lz.min(sz - 1);
            Self::is_transparent(self.blocks[flat_idx(lx, ly, lz)], water_block_id, registry)
        } else {
            let b = get_neighbor(
                wx + world_offset.0,
                wy + world_offset.1,
                wz + world_offset.2,
            );
            Self::is_transparent(b, water_block_id, registry)
        }
    }

    // -----------------------------------------------------------------------
    // Save / load DTO helpers
    // -----------------------------------------------------------------------

    /// Serialise this chunk into a compact DTO for disk storage.
    pub fn to_save_data(&self) -> crate::core::save_system::SavedChunk {
        let fluid_sparse: Vec<(u32, u8)> = self.fluid_levels
            .iter()
            .enumerate()
            .filter(|(_, v)| **v > 0)
            .map(|(i, &v)| (i as u32, v))
            .collect();

        crate::core::save_system::SavedChunk {
            cx: self.cx,
            cy: self.cy,
            cz: self.cz,
            blocks:       self.blocks.clone(),
            rotations:    self.rotations.clone(),
            fluid_sparse,
        }
    }

    /// Overwrite this chunk's block/rotation/fluid data from a saved DTO.
    /// Call after terrain generation to apply player edits over generated terrain.
    pub fn apply_save_data(&mut self, data: &crate::core::save_system::SavedChunk) {
        let vol = self.blocks.len();
        if data.blocks.len() == vol {
            self.blocks.copy_from_slice(&data.blocks);
        }
        if data.rotations.len() == vol {
            self.rotations.copy_from_slice(&data.rotations);
        }
        for v in self.fluid_levels.iter_mut() { *v = 0; }
        for &(idx, level) in &data.fluid_sparse {
            if (idx as usize) < vol {
                self.fluid_levels[idx as usize] = level;
            }
        }
        self.mesh_dirty = true;
    }

    // -----------------------------------------------------------------------
    // Slab mesh builder — half-block (0.5 height) in any rotation
    // -----------------------------------------------------------------------

    /// Build a mesh for all Slab blocks in this chunk.
    ///
    /// A Slab is a solid half-block.  Rotation 0 = floor slab (bottom half),
    /// rotation 5 = ceiling slab (top half).  Wall slabs use rotations 6–9.
    /// Rendered in the opaque pass (same pipeline as cube blocks).
    pub fn build_slab_mesh(
        &self,
        registry: &BlockRegistry,
        atlas: &TextureAtlasLayout,
    ) -> (Vec<Vertex3D>, Vec<u32>) {
        let sx = csx();
        let sy = csy();
        let sz = csz();

        let wx_base = self.cx * sx as i32;
        let wy_base = self.cy * sy as i32;
        let wz_base = self.cz * sz as i32;

        let mut vertices: Vec<Vertex3D> = Vec::new();
        let mut indices:  Vec<u32>      = Vec::new();

        for bx in 0..sx {
            for by in 0..sy {
                for bz in 0..sz {
                    let block_id = self.blocks[flat_idx(bx, by, bz)];
                    if block_id == 0 { continue; }
                    let def = match registry.get(block_id) {
                        Some(d) => d,
                        None => continue,
                    };
                    if def.block_shape != BlockShape::Slab { continue; }

                    let ox = wx_base as f32 + bx as f32;
                    let oy = wy_base as f32 + by as f32;
                    let oz = wz_base as f32 + bz as f32;

                    let rot = self.get_rotation(bx, by, bz);

                    let mat = &def.material;
                    let material_data = [mat.roughness, mat.metalness];
                    let emi = &def.emission;
                    let emission_data = if emi.emit_light {
                        [emi.light_color[0], emi.light_color[1], emi.light_color[2], -emi.light_intensity.abs()]
                    } else {
                        [0.0f32; 4]
                    };

                    // Determine slab geometry based on rotation:
                    //   rot 0 / 2 (floor slab): y 0..0.5
                    //   rot 5     (ceil slab):  y 0.5..1
                    //   rot 6 (wall N): z 0..0.5
                    //   rot 7 (wall S): z 0.5..1
                    //   rot 8 (wall E): x 0.5..1
                    //   rot 9 (wall W): x 0..0.5
                    let (y_lo, y_hi, z_lo, z_hi, x_lo, x_hi) = match rot {
                        5            => (0.5, 1.0,  0.0, 1.0,  0.0, 1.0),
                        6            => (0.0, 1.0,  0.0, 0.5,  0.0, 1.0),
                        7            => (0.0, 1.0,  0.5, 1.0,  0.0, 1.0),
                        8            => (0.0, 1.0,  0.0, 1.0,  0.5, 1.0),
                        9            => (0.0, 1.0,  0.0, 1.0,  0.0, 0.5),
                        _ /*0,1,2…*/ => (0.0, 0.5,  0.0, 1.0,  0.0, 1.0),
                    };

                    // Emit 6 faces of the slab cuboid.
                    // Face directions: 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z
                    for face_dir in 0u8..6 {
                        let c = def.color_for_face(face_dir);
                        let uv = atlas.uv_for(def.texture_for_face(face_dir).unwrap_or(""));
                        emit_slab_face(
                            &mut vertices, &mut indices,
                            ox, oy, oz,
                            face_dir,
                            x_lo, x_hi, y_lo, y_hi, z_lo, z_hi,
                            c, uv, material_data, emission_data,
                        );
                    }
                }
            }
        }

        flow_debug_log!(
            "Chunk", "build_slab_mesh",
            "Chunk ({},{},{}) => {} slab verts, {} indices",
            self.cx, self.cy, self.cz, vertices.len(), indices.len()
        );

        (vertices, indices)
    }

    /// Returns the collision AABB (world-relative, in block-local [0,1] space)
    /// for a block given its shape and rotation.
    /// Used by the physics system to build partial compound shapes for Slabs.
    pub fn collision_aabb_for(shape: &BlockShape, rotation: u8) -> Option<([f32; 3], [f32; 3])> {
        match shape {
            BlockShape::Slab => {
                let (y_lo, y_hi, z_lo, z_hi, x_lo, x_hi) = match rotation {
                    5            => (0.5f32, 1.0, 0.0, 1.0, 0.0, 1.0),
                    6            => (0.0, 1.0, 0.0, 0.5, 0.0, 1.0),
                    7            => (0.0, 1.0, 0.5, 1.0, 0.0, 1.0),
                    8            => (0.0, 1.0, 0.0, 1.0, 0.5, 1.0),
                    9            => (0.0, 1.0, 0.0, 1.0, 0.0, 0.5),
                    _            => (0.0, 0.5, 0.0, 1.0, 0.0, 1.0),
                };
                Some(([x_lo, y_lo, z_lo], [x_hi, y_hi, z_hi]))
            }
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// emit_face_lod — emits a single face of size `scale × scale`
//
// All faces wound CCW from outside (FrontFace::Ccw + CullMode::Back).
// ---------------------------------------------------------------------------
fn emit_face_lod(
    verts: &mut Vec<Vertex3D>,
    idxs:  &mut Vec<u32>,
    ox: f32, oy: f32, oz: f32,
    dir:      u8,
    color:    [f32; 3],
    uv:       (f32, f32, f32, f32),
    scale:    f32,
    material: [f32; 2],   // [roughness, metalness]
    emission: [f32; 4],   // [r, g, b, intensity]
) {
    let base = verts.len() as u32;
    let s = scale;

    let (normal, corners): ([f32; 3], [[f32; 3]; 4]) = match dir {
        // +X
        0 => ([1.0, 0.0, 0.0], [
            [ox+s, oy,   oz+s],
            [ox+s, oy,   oz  ],
            [ox+s, oy+s, oz  ],
            [ox+s, oy+s, oz+s],
        ]),
        // -X
        1 => ([-1.0, 0.0, 0.0], [
            [ox, oy,   oz  ],
            [ox, oy,   oz+s],
            [ox, oy+s, oz+s],
            [ox, oy+s, oz  ],
        ]),
        // +Y (top)
        2 => ([0.0, 1.0, 0.0], [
            [ox,   oy+s, oz+s],
            [ox+s, oy+s, oz+s],
            [ox+s, oy+s, oz  ],
            [ox,   oy+s, oz  ],
        ]),
        // -Y (bottom)
        3 => ([0.0, -1.0, 0.0], [
            [ox,   oy, oz  ],
            [ox+s, oy, oz  ],
            [ox+s, oy, oz+s],
            [ox,   oy, oz+s],
        ]),
        // +Z
        4 => ([0.0, 0.0, 1.0], [
            [ox,   oy,   oz+s],
            [ox+s, oy,   oz+s],
            [ox+s, oy+s, oz+s],
            [ox,   oy+s, oz+s],
        ]),
        // -Z
        _ => ([0.0, 0.0, -1.0], [
            [ox+s, oy,   oz],
            [ox,   oy,   oz],
            [ox,   oy+s, oz],
            [ox+s, oy+s, oz],
        ]),
    };

    let tangent: [f32; 3] = match dir {
        0 => [0.0,  0.0, -1.0],
        1 => [0.0,  0.0,  1.0],
        2 => [1.0,  0.0,  0.0],
        3 => [1.0,  0.0,  0.0],
        4 => [1.0,  0.0,  0.0],
        _ => [-1.0, 0.0,  0.0],
    };

    let (u0, v0, u1, v1) = uv;
    // UV per face — v1 goes to bottom vertices, v0 to top vertices
    // so that texture top (v0, grass belt) maps to block top.
    // U is mirrored for ±Z faces to maintain consistent left/right orientation.
    let uvs: [[f32; 2]; 4] = match dir {
        // +Y (top)
        2 => [ [u0, v1], [u1, v1], [u1, v0], [u0, v0] ],
        // -Y (bottom)
        3 => [ [u0, v0], [u1, v0], [u1, v1], [u0, v1] ],
        // +X (East) — was falling into `_` with flipped V
        0 => [ [u0, v1], [u1, v1], [u1, v0], [u0, v0] ],
        // -X (West)
        1 => [ [u0, v1], [u1, v1], [u1, v0], [u0, v0] ],
        // +Z (South)
        4 => [ [u1, v1], [u0, v1], [u0, v0], [u1, v0] ],
        // -Z (North) — was falling into `_` with flipped V
        5 => [ [u1, v1], [u0, v1], [u0, v0], [u1, v0] ],
        // fallback (should not be reached)
        _  => [ [u0, v1], [u1, v1], [u1, v0], [u0, v0] ],
    };

    for i in 0..4 {
        verts.push(Vertex3D {
            position:  corners[i],
            normal:    [snorm8(normal[0]), snorm8(normal[1]), snorm8(normal[2]), 0],
            color_ao:  [unorm8(color[0]),  unorm8(color[1]),  unorm8(color[2]),  255],
            texcoord:  uvs[i],
            material:  [unorm8(material[0]), unorm8(material[1]), 0, 0],
            emission,
            tangent:   [snorm8(tangent[0]), snorm8(tangent[1]), snorm8(tangent[2]), 0],
        });
    }
    idxs.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
}

/// Same as `emit_face_lod` but writes a custom alpha into `color_ao[3]`.
/// Used for transparent solid blocks (glass) so the fragment shader can
/// multiply texture alpha × vertex alpha to compute final opacity.
fn emit_face_lod_with_alpha(
    verts: &mut Vec<Vertex3D>,
    idxs:  &mut Vec<u32>,
    ox: f32, oy: f32, oz: f32,
    dir:      u8,
    color:    [f32; 3],
    alpha:    u8,
    uv:       (f32, f32, f32, f32),
    scale:    f32,
    material: [f32; 2],
    emission: [f32; 4],
) {
    let base = verts.len() as u32;
    let s = scale;

    let (normal, corners): ([f32; 3], [[f32; 3]; 4]) = match dir {
        0 => ([1.0, 0.0, 0.0],  [[ox+s,oy,oz+s],[ox+s,oy,oz],[ox+s,oy+s,oz],[ox+s,oy+s,oz+s]]),
        1 => ([-1.0, 0.0, 0.0], [[ox,oy,oz],[ox,oy,oz+s],[ox,oy+s,oz+s],[ox,oy+s,oz]]),
        2 => ([0.0, 1.0, 0.0],  [[ox,oy+s,oz+s],[ox+s,oy+s,oz+s],[ox+s,oy+s,oz],[ox,oy+s,oz]]),
        3 => ([0.0,-1.0, 0.0],  [[ox,oy,oz],[ox+s,oy,oz],[ox+s,oy,oz+s],[ox,oy,oz+s]]),
        4 => ([0.0, 0.0, 1.0],  [[ox,oy,oz+s],[ox+s,oy,oz+s],[ox+s,oy+s,oz+s],[ox,oy+s,oz+s]]),
        _ => ([0.0, 0.0,-1.0],  [[ox+s,oy,oz],[ox,oy,oz],[ox,oy+s,oz],[ox+s,oy+s,oz]]),
    };
    let tangent: [f32; 3] = match dir {
        0 => [0.0, 0.0,-1.0], 1 => [0.0, 0.0, 1.0],
        2 => [1.0, 0.0, 0.0], 3 => [1.0, 0.0, 0.0],
        4 => [1.0, 0.0, 0.0], _ => [-1.0, 0.0, 0.0],
    };
    let (u0, v0, u1, v1) = uv;
    let uvs: [[f32; 2]; 4] = match dir {
        2 => [[u0,v1],[u1,v1],[u1,v0],[u0,v0]],
        3 => [[u0,v0],[u1,v0],[u1,v1],[u0,v1]],
        0 => [[u0,v1],[u1,v1],[u1,v0],[u0,v0]],
        1 => [[u0,v1],[u1,v1],[u1,v0],[u0,v0]],
        4 => [[u1,v1],[u0,v1],[u0,v0],[u1,v0]],
        5 => [[u1,v1],[u0,v1],[u0,v0],[u1,v0]],
        _ => [[u0,v1],[u1,v1],[u1,v0],[u0,v0]],
    };

    for i in 0..4 {
        verts.push(Vertex3D {
            position:  corners[i],
            normal:    [snorm8(normal[0]), snorm8(normal[1]), snorm8(normal[2]), 0],
            color_ao:  [unorm8(color[0]),  unorm8(color[1]),  unorm8(color[2]),  alpha],
            texcoord:  uvs[i],
            material:  [unorm8(material[0]), unorm8(material[1]), 0, 0],
            emission,
            // tangent.w = 1 flags this vertex for the alpha-tested/blended path:
            // the fragment shader derives final opacity from texture.a × color_ao.w
            // (the per-block glass opacity). Opaque faces leave tangent.w = 0.
            tangent:   [snorm8(tangent[0]), snorm8(tangent[1]), snorm8(tangent[2]), snorm8(1.0)],
        });
    }
    idxs.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
}

// ---------------------------------------------------------------------------
// emit_water_face — emits a single water face with variable height
//
// `h` is the water level (0..1) controlling the top surface height.
// `alpha` controls transparency (stored in color_ao[3]).
// ---------------------------------------------------------------------------
fn emit_water_face(
    verts: &mut Vec<Vertex3D>,
    idxs:  &mut Vec<u32>,
    ox: f32, oy: f32, oz: f32,
    dir:      u8,
    // Per-corner surface heights (0..1), named by their XZ position:
    //   h_mm = (-x, -z)  h_pm = (+x, -z)
    //   h_mp = (-x, +z)  h_pp = (+x, +z)
    h_mm: f32, h_pm: f32, h_pp: f32, h_mp: f32,
    color:    [f32; 3],
    alpha:    u8,        // transparency (0..255)
    uv:       (f32, f32, f32, f32),
    material: [f32; 2],  // [roughness, metalness]
    emission: [f32; 4],  // [r, g, b, intensity] for luminous fluids
) {
    let base = verts.len() as u32;

    // Each side face's top edge uses the two corners on that face.
    // This makes the side face connect seamlessly with the top face.
    let (normal, corners): ([f32; 3], [[f32; 3]; 4]) = match dir {
        // +X face: top corners are (+x,-z)=h_pm and (+x,+z)=h_pp
        0 => ([1.0, 0.0, 0.0], [
            [ox+1.0, oy,        oz+1.0],
            [ox+1.0, oy,        oz     ],
            [ox+1.0, oy+h_pm,   oz     ],
            [ox+1.0, oy+h_pp,   oz+1.0],
        ]),
        // -X face: top corners are (-x,+z)=h_mp and (-x,-z)=h_mm
        1 => ([-1.0, 0.0, 0.0], [
            [ox, oy,        oz     ],
            [ox, oy,        oz+1.0 ],
            [ox, oy+h_mp,   oz+1.0 ],
            [ox, oy+h_mm,   oz     ],
        ]),
        // +Y (top surface): per-corner heights, counter-clockwise from (-x,+z)
        2 => ([0.0, 1.0, 0.0], [
            [ox,      oy+h_mp, oz+1.0],
            [ox+1.0,  oy+h_pp, oz+1.0],
            [ox+1.0,  oy+h_pm, oz     ],
            [ox,      oy+h_mm, oz     ],
        ]),
        // -Y (bottom): flat
        3 => ([0.0, -1.0, 0.0], [
            [ox,      oy, oz     ],
            [ox+1.0,  oy, oz     ],
            [ox+1.0,  oy, oz+1.0],
            [ox,      oy, oz+1.0],
        ]),
        // +Z face: top corners are (+x,+z)=h_pp and (-x,+z)=h_mp
        4 => ([0.0, 0.0, 1.0], [
            [ox,      oy,        oz+1.0],
            [ox+1.0,  oy,        oz+1.0],
            [ox+1.0,  oy+h_pp,   oz+1.0],
            [ox,      oy+h_mp,   oz+1.0],
        ]),
        // -Z face: top corners are (-x,-z)=h_mm and (+x,-z)=h_pm
        _ => ([0.0, 0.0, -1.0], [
            [ox+1.0,  oy,        oz],
            [ox,      oy,        oz],
            [ox,      oy+h_mm,   oz],
            [ox+1.0,  oy+h_pm,   oz],
        ]),
    };

    let tangent: [f32; 3] = match dir {
        0 => [0.0,  0.0, -1.0],
        1 => [0.0,  0.0,  1.0],
        2 => [1.0,  0.0,  0.0],
        3 => [1.0,  0.0,  0.0],
        4 => [1.0,  0.0,  0.0],
        _ => [-1.0, 0.0,  0.0],
    };

    let (u0, v0, u1, v1) = uv;
    let uvs: [[f32; 2]; 4] = match dir {
        2 => [ [u0, v1], [u1, v1], [u1, v0], [u0, v0] ],
        3 => [ [u0, v0], [u1, v0], [u1, v1], [u0, v1] ],
        0 => [ [u0, v1], [u1, v1], [u1, v0], [u0, v0] ],
        1 => [ [u0, v1], [u1, v1], [u1, v0], [u0, v0] ],
        4 => [ [u1, v1], [u0, v1], [u0, v0], [u1, v0] ],
        5 => [ [u1, v1], [u0, v1], [u0, v0], [u1, v0] ],
        _  => [ [u0, v1], [u1, v1], [u1, v0], [u0, v0] ],
    };

    for i in 0..4 {
        verts.push(Vertex3D {
            position:  corners[i],
            normal:    [snorm8(normal[0]), snorm8(normal[1]), snorm8(normal[2]), 0],
            color_ao:  [unorm8(color[0]),  unorm8(color[1]),  unorm8(color[2]),  alpha],
            texcoord:  uvs[i],
            material:  [unorm8(material[0]), unorm8(material[1]), 0, 0],
            emission,
            tangent:   [snorm8(tangent[0]), snorm8(tangent[1]), snorm8(tangent[2]), 0],
        });
    }
    idxs.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
}

// ---------------------------------------------------------------------------
// emit_foliage_quad — emits one double-sided quad for foliage (Cross/CropGrid)
// ---------------------------------------------------------------------------
fn emit_foliage_quad(
    verts:    &mut Vec<Vertex3D>,
    idxs:     &mut Vec<u32>,
    corners:  [[f32; 3]; 4],
    color:    [f32; 3],
    uv:       (f32, f32, f32, f32),
    material: [f32; 2],
    emission: [f32; 4],
) {
    let base = verts.len() as u32;

    let e1 = [corners[1][0]-corners[0][0], corners[1][1]-corners[0][1], corners[1][2]-corners[0][2]];
    let e2 = [corners[3][0]-corners[0][0], corners[3][1]-corners[0][1], corners[3][2]-corners[0][2]];
    let nx = e1[1]*e2[2] - e1[2]*e2[1];
    let ny = e1[2]*e2[0] - e1[0]*e2[2];
    let nz = e1[0]*e2[1] - e1[1]*e2[0];
    let nlen = (nx*nx + ny*ny + nz*nz).sqrt().max(1e-6);
    let normal = [nx/nlen, ny/nlen, nz/nlen];
    let e1n = [e1[0]/nlen, e1[1]/nlen, e1[2]/nlen];

    let (u0, v0, u1, v1) = uv;
    let uvs: [[f32; 2]; 4] = [
        [u0, v1], [u1, v1], [u1, v0], [u0, v0],
    ];

    for i in 0..4 {
        verts.push(Vertex3D {
            position:  corners[i],
            normal:    [snorm8(normal[0]), snorm8(normal[1]), snorm8(normal[2]), 0],
            color_ao:  [unorm8(color[0]),  unorm8(color[1]),  unorm8(color[2]),  255],
            texcoord:  uvs[i],
            material:  [unorm8(material[0]), unorm8(material[1]), 0, 0],
            emission,
            tangent:   [snorm8(e1n[0]), snorm8(e1n[1]), snorm8(e1n[2]), 0],
        });
    }
    // Front face (CCW) + back face (CW) for double-sided rendering
    idxs.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
    idxs.extend_from_slice(&[base, base+2, base+1, base, base+3, base+2]);
}

// ---------------------------------------------------------------------------
// emit_slab_face — emits one face of a slab cuboid
// dir: 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z
// ---------------------------------------------------------------------------
fn emit_slab_face(
    verts:    &mut Vec<Vertex3D>,
    idxs:     &mut Vec<u32>,
    ox: f32, oy: f32, oz: f32,
    dir:      u8,
    x_lo: f32, x_hi: f32,
    y_lo: f32, y_hi: f32,
    z_lo: f32, z_hi: f32,
    color:    [f32; 3],
    uv:       (f32, f32, f32, f32),
    material: [f32; 2],
    emission: [f32; 4],
) {
    let base = verts.len() as u32;

    let (x0, x1) = (ox + x_lo, ox + x_hi);
    let (y0, y1) = (oy + y_lo, oy + y_hi);
    let (z0, z1) = (oz + z_lo, oz + z_hi);

    let (normal, corners): ([f32; 3], [[f32; 3]; 4]) = match dir {
        0 => ([1.0, 0.0, 0.0],  [[x1,y0,z1],[x1,y0,z0],[x1,y1,z0],[x1,y1,z1]]),
        1 => ([-1.0, 0.0, 0.0], [[x0,y0,z0],[x0,y0,z1],[x0,y1,z1],[x0,y1,z0]]),
        2 => ([0.0, 1.0, 0.0],  [[x0,y1,z1],[x1,y1,z1],[x1,y1,z0],[x0,y1,z0]]),
        3 => ([0.0,-1.0, 0.0],  [[x0,y0,z0],[x1,y0,z0],[x1,y0,z1],[x0,y0,z1]]),
        4 => ([0.0, 0.0, 1.0],  [[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]]),
        _ => ([0.0, 0.0,-1.0],  [[x1,y0,z0],[x0,y0,z0],[x0,y1,z0],[x1,y1,z0]]),
    };
    let tangent: [f32; 3] = match dir {
        0 => [0.0, 0.0,-1.0], 1 => [0.0, 0.0, 1.0],
        2 => [1.0, 0.0, 0.0], 3 => [1.0, 0.0, 0.0],
        4 => [1.0, 0.0, 0.0], _ => [-1.0, 0.0, 0.0],
    };
    let (u0, v0, u1, v1) = uv;
    let uvs: [[f32; 2]; 4] = match dir {
        2 => [[u0,v1],[u1,v1],[u1,v0],[u0,v0]],
        3 => [[u0,v0],[u1,v0],[u1,v1],[u0,v1]],
        4 => [[u1,v1],[u0,v1],[u0,v0],[u1,v0]],
        5 => [[u1,v1],[u0,v1],[u0,v0],[u1,v0]],
        _ => [[u0,v1],[u1,v1],[u1,v0],[u0,v0]],
    };

    for i in 0..4 {
        verts.push(Vertex3D {
            position:  corners[i],
            normal:    [snorm8(normal[0]), snorm8(normal[1]), snorm8(normal[2]), 0],
            color_ao:  [unorm8(color[0]),  unorm8(color[1]),  unorm8(color[2]),  255],
            texcoord:  uvs[i],
            material:  [unorm8(material[0]), unorm8(material[1]), 0, 0],
            emission,
            tangent:   [snorm8(tangent[0]), snorm8(tangent[1]), snorm8(tangent[2]), 0],
        });
    }
    idxs.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
}
