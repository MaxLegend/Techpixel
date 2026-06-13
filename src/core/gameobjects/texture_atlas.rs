// =============================================================================
// QubePixel — TextureAtlas  (albedo + normal map, stitched atlas, mip-mapped)
// =============================================================================
//
// CHANGELOG:
//   • Mip-mapping: generates per-tile mip levels (box filter) to prevent
//     tile bleeding at atlas seams.  Uses trilinear-ready mip chain.
//   • mip_level_count() exposed for sampler configuration.
//   • Normal map atlas: parallel atlas loaded from *_n.png files,
//     falls back to flat normals (128,128,255) per tile.
// =============================================================================

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::collections::HashSet;
use image::{imageops, Rgba, RgbaImage};
use crate::debug_log;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
pub const TILE_SIZE:    u32 = 16;
pub const ATLAS_COLS:   u32 = 64;
pub const ATLAS_ROWS:   u32 = 64;
/// Tracks texture keys that have already triggered a "not found" warning.
/// Each unique missing key is logged exactly once to avoid log spam.
static MISSING_UV_WARNINGS: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
pub const ATLAS_WIDTH:  u32 = TILE_SIZE * ATLAS_COLS; // 256
pub const ATLAS_HEIGHT: u32 = TILE_SIZE * ATLAS_ROWS; // 256

const TEXTURES_DIR: &str = "assets/textures/simpleblocks";

/// Number of mip levels: log2(TILE_SIZE) + 1  →  log2(16) + 1 = 5
/// Levels: 256×256 → 128×128 → 64×64 → 32×32 → 16×16
pub const MIP_LEVELS: u32 = 5;

// ---------------------------------------------------------------------------
// TextureAtlasLayout
// ---------------------------------------------------------------------------
#[derive(Clone, Debug)]
pub struct TextureAtlasLayout {
    pub(crate) uv_map: HashMap<String, (f32, f32, f32, f32)>,
}

impl TextureAtlasLayout {
    pub fn uv_for(&self, name: &str) -> (f32, f32, f32, f32) {
        if let Some(uv) = self.uv_map.get(name) {
            return *uv;
        }
        // Log warning once per unique missing key to help diagnose atlas issues.
        let warnings = MISSING_UV_WARNINGS.get_or_init(|| Mutex::new(HashSet::new()));
        if let Ok(mut set) = warnings.lock() {
            if set.insert(name.to_string()) {
                debug_log!(
                    "TextureAtlasLayout", "uv_for",
                    "WARNING: texture key '{}' not found in atlas ({} keys loaded). \
                     Check that the PNG file exists under assets/textures/ and the JSON \
                     'texture' field matches the relative path (e.g. 'rocks/andesite' \
                     for assets/textures/rocks/andesite.png)",
                    name,
                    self.uv_map.len()
                );
            }
        }
        Self::white_uv()
    }
    /// Dump all registered UV keys to the log. Call once after atlas loading
    /// to verify that texture names match block JSON definitions.
    pub fn dump_keys(&self) {
        let mut keys: Vec<&String> = self.uv_map.keys().collect();
        keys.sort();
        debug_log!(
            "TextureAtlasLayout", "dump_keys",
            "Atlas contains {} texture keys:", keys.len()
        );
        for k in &keys {
            let (u0, v0, u1, v1) = self.uv_map[*k];
            debug_log!(
                "TextureAtlasLayout", "dump_keys",
                "  '{}' => UV ({:.3},{:.3})-({:.3},{:.3})", k, u0, v0, u1, v1
            );
        }
    }
    fn white_uv() -> (f32, f32, f32, f32) {
        let u1 = 1.0 / ATLAS_COLS as f32;
        let v1 = 1.0 / ATLAS_ROWS as f32;
        (0.0, 0.0, u1, v1)
    }

}

impl Default for TextureAtlasLayout {
    fn default() -> Self { Self { uv_map: HashMap::new() } }
}

// ---------------------------------------------------------------------------
// TextureAtlas
// ---------------------------------------------------------------------------
pub struct TextureAtlas {
    /// Base albedo atlas image (mip level 0).
    image:       RgbaImage,
    /// Pre-computed albedo mip levels 1..N (level 0 = self.image).
    mip_images:  Vec<RgbaImage>,
    /// Normal map atlas image (mip level 0). Same slot layout as albedo.
    normal_image:       RgbaImage,
    /// Pre-computed normal mip levels 1..N.
    normal_mip_images:  Vec<RgbaImage>,
    layout:      TextureAtlasLayout,
    // True when baked textures have been added and mip chain needs rebuild.
    // Checked in `ensure_gpu()` — auto-regenerates before GPU upload.
    mip_dirty:   bool,
    // -- Albedo GPU --
    gpu_texture: Option<wgpu::Texture>,
    gpu_view:    Option<wgpu::TextureView>,
    // -- Normal GPU --
    normal_gpu_texture: Option<wgpu::Texture>,
    normal_gpu_view:    Option<wgpu::TextureView>,
}

impl TextureAtlas {
    pub fn load() -> Self { Self::load_from_dir(TEXTURES_DIR) }

    pub fn load_from_dir(dir: &str) -> Self {
        let path = PathBuf::from(dir);
        if path.is_dir() {
            debug_log!("TextureAtlas", "load", "Loading textures from {:?}", path);
            match Self::build_from_dir(&path) {
                Ok(atlas) => return atlas,
                Err(e)    => debug_log!(
                    "TextureAtlas", "load",
                    "Directory load failed: {}, using embedded fallback", e
                ),
            }
        } else {
            debug_log!(
                "TextureAtlas", "load",
                "Directory {:?} not found, using embedded fallback", path
            );
        }
        Self::build_embedded()
    }

    pub fn layout(&self) -> &TextureAtlasLayout { &self.layout }

    /// Inspect the named atlas tile's alpha channel.
    /// Returns `(has_alpha, avg_opacity)`:
    ///   • `has_alpha` — any texel is non-opaque (alpha < 200), i.e. a hole or a
    ///     semi-transparent pixel. The 200 threshold ignores anti-aliased edges.
    ///   • `avg_opacity` — mean alpha over the tile, normalised to 0..1. Drives how
    ///     much a texture-transparent block dims a shadow ray (1 = fully opaque).
    /// A missing texture reports `(false, 1.0)` (treated as opaque).
    pub fn texture_alpha_info(&self, name: &str) -> (bool, f32) {
        match self.tile_rgba(name) {
            Some(px) => {
                let mut sum = 0u32;
                let mut n   = 0u32;
                let mut has = false;
                for p in px.chunks_exact(4) {
                    let a = p[3];
                    sum += a as u32;
                    n   += 1;
                    if a < 200 { has = true; }
                }
                let avg = if n > 0 { (sum as f32 / n as f32) / 255.0 } else { 1.0 };
                (has, avg)
            }
            None => (false, 1.0),
        }
    }

    /// Extract the 16×16 RGBA tile for `name` from the CPU-side atlas image.
    /// Returns `None` if the texture name is not registered.
    pub fn tile_rgba(&self, name: &str) -> Option<Vec<u8>> {
        let uv = self.layout.uv_map.get(name)?;
        let px = (uv.0 * ATLAS_WIDTH  as f32).round() as u32;
        let py = (uv.1 * ATLAS_HEIGHT as f32).round() as u32;
        let sub = image::imageops::crop_imm(&self.image, px, py, TILE_SIZE, TILE_SIZE);
        Some(sub.to_image().into_raw())
    }

    /// Upload albedo + normal atlas + mip chains to GPU.
    pub fn ensure_gpu(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.gpu_texture.is_some() { return; }

        // Auto-regenerate mip chain if baked textures were added after load
        if self.mip_dirty {
            self.regenerate_mips();
            self.mip_dirty = false;
        }

        debug_log!(
            "TextureAtlas", "ensure_gpu",
            "Uploading albedo+normal atlas {}x{} with {} mip levels to GPU",
            ATLAS_WIDTH, ATLAS_HEIGHT, MIP_LEVELS
        );

        // ---- Albedo atlas ----
        let albedo_texture = device.create_texture(&wgpu::TextureDescriptor {
            label:           Some("TextureAtlas"),
            size:            wgpu::Extent3d {
                width: ATLAS_WIDTH,
                height: ATLAS_HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: MIP_LEVELS,
            sample_count:    1,
            dimension:       wgpu::TextureDimension::D2,
            format:          wgpu::TextureFormat::Rgba8Unorm,
            usage:           wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats:    &[],
        });

        // Upload albedo mip level 0
        let w0 = ATLAS_WIDTH;
        let h0 = ATLAS_HEIGHT;
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture:   &albedo_texture,
                mip_level: 0,
                origin:    wgpu::Origin3d::ZERO,
                aspect:    wgpu::TextureAspect::All,
            },
            &self.image,
            wgpu::TexelCopyBufferLayout {
                offset:         0,
                bytes_per_row:  Some(w0 * 4),
                rows_per_image: Some(h0),
            },
            wgpu::Extent3d { width: w0, height: h0, depth_or_array_layers: 1 },
        );

        // Upload albedo mip levels 1..N
        for (i, mip) in self.mip_images.iter().enumerate() {
            let level = (i + 1) as u32;
            let w = ATLAS_WIDTH >> level;
            let h = ATLAS_HEIGHT >> level;
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture:   &albedo_texture,
                    mip_level: level,
                    origin:    wgpu::Origin3d::ZERO,
                    aspect:    wgpu::TextureAspect::All,
                },
                mip,
                wgpu::TexelCopyBufferLayout {
                    offset:         0,
                    bytes_per_row:  Some(w * 4),
                    rows_per_image: Some(h),
                },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
            debug_log!(
                "TextureAtlas", "ensure_gpu",
                "Albedo mip level {} ({}x{})", level, w, h
            );
        }

        let albedo_view = albedo_texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.gpu_texture = Some(albedo_texture);
        self.gpu_view    = Some(albedo_view);

        // ---- Normal map atlas ----
        let normal_texture = device.create_texture(&wgpu::TextureDescriptor {
            label:           Some("NormalAtlas"),
            size:            wgpu::Extent3d {
                width: ATLAS_WIDTH,
                height: ATLAS_HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: MIP_LEVELS,
            sample_count:    1,
            dimension:       wgpu::TextureDimension::D2,
            format:          wgpu::TextureFormat::Rgba8Unorm,
            usage:           wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats:    &[],
        });

        // Upload normal mip level 0
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture:   &normal_texture,
                mip_level: 0,
                origin:    wgpu::Origin3d::ZERO,
                aspect:    wgpu::TextureAspect::All,
            },
            &self.normal_image,
            wgpu::TexelCopyBufferLayout {
                offset:         0,
                bytes_per_row:  Some(ATLAS_WIDTH * 4),
                rows_per_image: Some(ATLAS_HEIGHT),
            },
            wgpu::Extent3d { width: ATLAS_WIDTH, height: ATLAS_HEIGHT, depth_or_array_layers: 1 },
        );

        // Upload normal mip levels 1..N
        for (i, mip) in self.normal_mip_images.iter().enumerate() {
            let level = (i + 1) as u32;
            let w = ATLAS_WIDTH >> level;
            let h = ATLAS_HEIGHT >> level;
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture:   &normal_texture,
                    mip_level: level,
                    origin:    wgpu::Origin3d::ZERO,
                    aspect:    wgpu::TextureAspect::All,
                },
                mip,
                wgpu::TexelCopyBufferLayout {
                    offset:         0,
                    bytes_per_row:  Some(w * 4),
                    rows_per_image: Some(h),
                },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
            debug_log!(
                "TextureAtlas", "ensure_gpu",
                "Normal mip level {} ({}x{})", level, w, h
            );
        }

        let normal_view = normal_texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.normal_gpu_texture = Some(normal_texture);
        self.normal_gpu_view    = Some(normal_view);

        debug_log!("TextureAtlas", "ensure_gpu", "Albedo + normal atlas uploaded with mip chains");
    }

    pub fn texture_view(&self) -> Option<&wgpu::TextureView> { self.gpu_view.as_ref() }

    /// Normal map atlas view (same slot layout as albedo atlas).
    pub fn normal_texture_view(&self) -> Option<&wgpu::TextureView> { self.normal_gpu_view.as_ref() }


    // -----------------------------------------------------------------------
    // Layer baking — alpha-blended texture compositing
    // -----------------------------------------------------------------------

    /// Bake a layered texture: composite `layer_keys` on top of `base_key`
    /// using Porter-Duff "source over" alpha blending.
    ///
    /// Returns the deterministic baked texture key (added to the atlas UV
    /// map), or `None` if the base texture is not found or the atlas is full.
    ///
    /// The caller (typically `BlockRegistry::bake_layered_textures`) stores
    /// the returned key in `FaceOverride::resolved_texture`.
    ///
    /// **Transparency handling:**
    ///   - Layer pixels with alpha = 0 are completely skipped.
    ///   - Semi-transparent pixels are blended proportionally.
    ///   - Fully opaque layer pixels replace the base pixel.
    ///   - The result's alpha is clamped to 255.
    pub fn bake_layered_texture(
        &mut self,
        base_key: &str,
        layer_keys: &[&str],
    ) -> Option<String> {
        // --- 1. Look up base tile in atlas ---
        let (base_u0, base_v0, _, _) = *self.layout.uv_map.get(base_key)?;

        // --- 2. Build deterministic baked key ---
        // Format: __baked__<base>+<layer1>+<layer2>+...
        // Same combination always produces the same key (deduplication).
        let baked_key = format!(
            "__baked__{}",
            layer_keys.iter().fold(base_key.to_string(), |acc, l| format!("{}+{}", acc, l))
        );

        // Already baked (another block uses the same combination)?
        if self.layout.uv_map.contains_key(&baked_key) {
            debug_log!(
                "TextureAtlas", "bake_layered_texture",
                "Already baked: '{}'", baked_key
            );
            return Some(baked_key);
        }

        // --- 3. Read base tile pixels ---
        let px0 = (base_u0 * ATLAS_WIDTH as f32) as u32;
        let py0 = (base_v0 * ATLAS_HEIGHT as f32) as u32;

        let mut pixels: Vec<[u8; 4]> = Vec::with_capacity((TILE_SIZE * TILE_SIZE) as usize);
        for y in 0..TILE_SIZE {
            for x in 0..TILE_SIZE {
                let p = self.image.get_pixel(px0 + x, py0 + y);
                pixels.push([p[0], p[1], p[2], p[3]]);
            }
        }

        // --- 4. Apply each layer with alpha blending ---
        for layer_key in layer_keys {
            let (lu0, lv0, _, _) = match self.layout.uv_map.get(*layer_key) {
                Some(uv) => *uv,
                None => {
                    debug_log!(
                        "TextureAtlas", "bake_layered_texture",
                        "WARNING: layer texture '{}' not found in atlas, skipping",
                        layer_key
                    );
                    continue;
                }
            };

            let lpx0 = (lu0 * ATLAS_WIDTH as f32) as u32;
            let lpy0 = (lv0 * ATLAS_HEIGHT as f32) as u32;

            for y in 0..TILE_SIZE {
                for x in 0..TILE_SIZE {
                    let layer_pixel = self.image.get_pixel(lpx0 + x, lpy0 + y);
                    let idx = (y * TILE_SIZE + x) as usize;
                    let base_pixel = &mut pixels[idx];

                    let a = layer_pixel[3] as f32 / 255.0;
                    if a < 0.001 {
                        continue; // Fully transparent — skip
                    }

                    // Porter-Duff "source over" alpha compositing
                    let inv_a = 1.0 - a;
                    base_pixel[0] = (base_pixel[0] as f32 * inv_a + layer_pixel[0] as f32 * a)
                        .round() as u8;
                    base_pixel[1] = (base_pixel[1] as f32 * inv_a + layer_pixel[1] as f32 * a)
                        .round() as u8;
                    base_pixel[2] = (base_pixel[2] as f32 * inv_a + layer_pixel[2] as f32 * a)
                        .round() as u8;
                    base_pixel[3] = ((base_pixel[3] as f32 * inv_a + 255.0 * a)
                        .round().min(255.0)) as u8;
                }
            }
        }

        // --- 5. Allocate a new slot in the atlas ---
        let (col, row) = match self.find_free_slot() {
            Some(slot) => slot,
            None => {
                debug_log!(
                    "TextureAtlas", "bake_layered_texture",
                    "WARNING: atlas full, cannot bake '{}'", baked_key
                );
                return None;
            }
        };

        // --- 6. Create baked tile image and place in atlas ---
        let mut tile = RgbaImage::new(TILE_SIZE, TILE_SIZE);
        for y in 0..TILE_SIZE {
            for x in 0..TILE_SIZE {
                let p = &pixels[(y * TILE_SIZE + x) as usize];
                tile.put_pixel(x, y, Rgba([p[0], p[1], p[2], p[3]]));
            }
        }

        // Albedo
        imageops::overlay(
            &mut self.image, &tile,
            (col * TILE_SIZE) as i64, (row * TILE_SIZE) as i64,
        );

        // Normal map: use flat normal for baked tiles
        // (dedicated normal baking is a future enhancement)
        fill_tile_flat_normal_at(&mut self.normal_image, col, row);

        // --- 7. Register UV ---
        let u0 = col as f32 / ATLAS_COLS as f32;
        let v0 = row as f32 / ATLAS_ROWS as f32;
        let u1 = (col + 1) as f32 / ATLAS_COLS as f32;
        let v1 = (row + 1) as f32 / ATLAS_ROWS as f32;
        self.layout.uv_map.insert(baked_key.clone(), (u0, v0, u1, v1));

        // Mark mip chain as dirty — will be regenerated in ensure_gpu()
        self.mip_dirty = true;

        debug_log!(
            "TextureAtlas", "bake_layered_texture",
            "Baked '{}' + {:?} => slot ({},{}) key='{}'",
            base_key, layer_keys, col, row, baked_key
        );

        Some(baked_key)
    }

    /// Find the first free (unused) tile slot in the atlas grid.
    /// Returns `(col, row)` or `None` if the atlas is completely full.
    fn find_free_slot(&self) -> Option<(u32, u32)> {
        // Collect all used slots from UV coordinates
        let mut used = std::collections::HashSet::new();
        for &(u0, v0, _, _) in self.layout.uv_map.values() {
            let col = (u0 * ATLAS_COLS as f32).round() as u32;
            let row = (v0 * ATLAS_ROWS as f32).round() as u32;
            used.insert((col, row));
        }

        // Linear scan for first free slot
        for row in 0..ATLAS_ROWS {
            for col in 0..ATLAS_COLS {
                if !used.contains(&(col, row)) {
                    return Some((col, row));
                }
            }
        }
        None
    }

    /// Regenerate the mip chain from the current atlas images.
    /// Called automatically by `ensure_gpu()` when `mip_dirty` is true,
    /// but can also be called manually after batch baking operations.
    pub fn regenerate_mips(&mut self) {
        debug_log!(
            "TextureAtlas", "regenerate_mips",
            "Regenerating mip chains ({}x{}, {} levels)",
            ATLAS_WIDTH, ATLAS_HEIGHT, MIP_LEVELS
        );
        self.mip_images = Self::generate_mip_chain(&self.image);
        self.normal_mip_images = Self::generate_mip_chain(&self.normal_image);
        debug_log!(
            "TextureAtlas", "regenerate_mips",
            "Mip chain regenerated: {} levels", self.mip_images.len() + 1
        );
    }
    // -----------------------------------------------------------------------
    // Mip generation — per-tile box filter (prevents atlas tile bleeding)
    // -----------------------------------------------------------------------

    /// Generate MIP_LEVELS−1 mip images from the base atlas.
    fn generate_mip_chain(base: &RgbaImage) -> Vec<RgbaImage> {
        let mut mips = Vec::with_capacity((MIP_LEVELS - 1) as usize);
        let mut prev = base.clone();
        let mut prev_tile_size = TILE_SIZE;

        for level in 1..MIP_LEVELS {
            let new_tile = prev_tile_size / 2;
            if new_tile == 0 { break; }

            let new_w = new_tile * ATLAS_COLS;
            let new_h = new_tile * ATLAS_ROWS;
            let mut mip = RgbaImage::new(new_w, new_h);

            for row in 0..ATLAS_ROWS {
                for col in 0..ATLAS_COLS {
                    for ty in 0..new_tile {
                        for tx in 0..new_tile {
                            let sx = col * prev_tile_size + tx * 2;
                            let sy = row * prev_tile_size + ty * 2;

                            let mut r = 0u32;
                            let mut g = 0u32;
                            let mut b = 0u32;
                            let mut a = 0u32;

                            for dy in 0..2u32 {
                                for dx in 0..2u32 {
                                    let px = (sx + dx).min(prev.width() - 1);
                                    let py = (sy + dy).min(prev.height() - 1);
                                    let p = prev.get_pixel(px, py);
                                    r += p[0] as u32;
                                    g += p[1] as u32;
                                    b += p[2] as u32;
                                    a += p[3] as u32;
                                }
                            }

                            let dst_x = col * new_tile + tx;
                            let dst_y = row * new_tile + ty;
                            mip.put_pixel(dst_x, dst_y, Rgba([
                                (r / 4) as u8,
                                (g / 4) as u8,
                                (b / 4) as u8,
                                (a / 4) as u8,
                            ]));
                        }
                    }
                }
            }

            debug_log!(
                "TextureAtlas", "generate_mip_chain",
                "Mip level {} : {}x{} (tile={}px)",
                level, new_w, new_h, new_tile
            );

            prev_tile_size = new_tile;
            prev = mip.clone();
            mips.push(mip);
        }

        mips
    }
    // -----------------------------------------------------------------------
    // Recursive texture collector — walks subdirectories, preserves relative path
    // -----------------------------------------------------------------------

    /// Recursively collects all `.png` texture files from `current` directory,
    /// computing a relative key path from `root`.
    ///
    /// Normal map files (`*_n.png`) are skipped — they are loaded separately
    /// when their matching albedo texture is processed.
    ///
    /// Each entry is `(full_path, relative_key)` where `relative_key` uses `/`
    /// as separator (e.g. `"rocks/andesite"`, `"grass/grass_top"`).
    fn collect_textures_recursive(
        root: &std::path::Path,
        current: &std::path::Path,
        out: &mut Vec<(PathBuf, String)>,
    ) -> Result<(), String> {
        let entries = std::fs::read_dir(current)
            .map_err(|e| format!("Cannot read dir {:?}: {}", current, e))?;

        for entry in entries {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };
            let path = entry.path();

            if path.is_dir() {
                // Recurse into subdirectory
                Self::collect_textures_recursive(root, &path, out)?;
                continue;
            }

            // Only .png files
            if !path.extension().map_or(false, |ext| ext == "png") {
                continue;
            }

            // Skip normal map files (*_n.png)
            let stem = path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            if stem.ends_with("_n") {
                continue;
            }

            // Compute relative path from root as the UV map key.
            // e.g. "assets/textures/rocks/andesite.png" → "rocks/andesite"
            let rel = path.strip_prefix(root)
                .map_err(|e| format!("strip_prefix failed: {}", e))?;
            let rel_str = rel.to_str()
                .ok_or_else(|| format!("non-UTF-8 path: {:?}", rel))?;
            let key = rel_str
                .strip_suffix(".png")
                .unwrap_or(rel_str)
                .replace('\\', "/")   // ← нормализуем для cross-platform
                .to_string();

            out.push((path, key));
        }

        Ok(())
    }
    // -----------------------------------------------------------------------
    // Directory loader
    // -----------------------------------------------------------------------

    fn build_from_dir(dir: &std::path::Path) -> Result<Self, String> {
        let mut atlas_img = RgbaImage::new(ATLAS_WIDTH, ATLAS_HEIGHT);
        let mut normal_img = RgbaImage::new(ATLAS_WIDTH, ATLAS_HEIGHT);
        let mut uv_map    = HashMap::new();

        fill_tile_solid(&mut atlas_img, 0, 0, 255, 255, 255);
        fill_tile_flat_normal(&mut normal_img, 0, 0);

        // --- Collect all .png textures recursively (excluding _n normal maps) ---
        let mut entries: Vec<(PathBuf, String)> = Vec::new();
        Self::collect_textures_recursive(dir, dir, &mut entries)?;

        // Sort by relative key for deterministic atlas layout
        entries.sort_by(|a, b| a.1.cmp(&b.1));

        if entries.is_empty() {
            debug_log!(
                "TextureAtlas", "build_from_dir",
                "WARNING: no textures found in {:?} (recursive)", dir
            );
        }

        let mut next_slot: u32 = 1;

        for (path, rel_key) in &entries {
            let stem = path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown");

            let dyn_img = image::open(path)
                .map_err(|e| format!("Failed to load {:?}: {}", path, e))?;
            let rgba = dyn_img.to_rgba8();

            let tile = if rgba.width() != TILE_SIZE || rgba.height() != TILE_SIZE {
                imageops::resize(&rgba, TILE_SIZE, TILE_SIZE, imageops::FilterType::Nearest)
            } else { rgba };

            let col = next_slot % ATLAS_COLS;
            let row = next_slot / ATLAS_COLS;
            if row >= ATLAS_ROWS {
                return Err(format!(
                    "Atlas full — {} textures exceed {} slots",
                    entries.len(),
                    (ATLAS_COLS * ATLAS_ROWS - 1) as usize
                ));
            }
            next_slot += 1;

            imageops::overlay(
                &mut atlas_img, &tile,
                (col * TILE_SIZE) as i64, (row * TILE_SIZE) as i64,
            );

            // --- Normal map: look for *_n.png in the SAME directory ---
            let parent_dir = path.parent().unwrap_or(dir);
            let normal_path = parent_dir.join(format!("{}_n.png", stem));
            if normal_path.exists() {
                match image::open(&normal_path) {
                    Ok(normal_dyn) => {
                        let normal_rgba = normal_dyn.to_rgba8();
                        let normal_tile = if normal_rgba.width() != TILE_SIZE
                            || normal_rgba.height() != TILE_SIZE
                        {
                            imageops::resize(&normal_rgba, TILE_SIZE, TILE_SIZE,
                                             imageops::FilterType::Nearest)
                        } else { normal_rgba };
                        imageops::overlay(
                            &mut normal_img, &normal_tile,
                            (col * TILE_SIZE) as i64, (row * TILE_SIZE) as i64,
                        );
                        debug_log!(
                            "TextureAtlas", "build_from_dir",
                            "Loaded normal map '{}' => slot ({},{})",
                            normal_path.display(), col, row
                        );
                    }
                    Err(e) => {
                        debug_log!(
                            "TextureAtlas", "build_from_dir",
                            "Failed to load normal '{:?}': {}, using flat",
                            normal_path, e
                        );
                        fill_tile_flat_normal_at(&mut normal_img, col, row);
                    }
                }
            } else {
                fill_tile_flat_normal_at(&mut normal_img, col, row);
            }

            let u0 = col as f32 / ATLAS_COLS as f32;
            let v0 = row as f32 / ATLAS_ROWS as f32;
            let u1 = (col + 1) as f32 / ATLAS_COLS as f32;
            let v1 = (row + 1) as f32 / ATLAS_ROWS as f32;

            debug_log!(
                "TextureAtlas", "build_from_dir",
                "Loaded '{}' => slot ({},{})", rel_key, col, row
            );
            uv_map.insert(rel_key.clone(), (u0, v0, u1, v1));
        }

        debug_log!(
            "TextureAtlas", "build_from_dir",
            "Total textures loaded (recursive): {}", uv_map.len()
        );
        // Diagnostic: dump all atlas keys so the user can verify texture names
        let layout_for_dump = TextureAtlasLayout { uv_map: uv_map.clone() };
        layout_for_dump.dump_keys();
        let mip_images = Self::generate_mip_chain(&atlas_img);
        let normal_mip_images = Self::generate_mip_chain(&normal_img);
        Ok(Self {
            image: atlas_img,
            mip_images,
            normal_image: normal_img,
            normal_mip_images,
            layout: TextureAtlasLayout { uv_map },
            mip_dirty: false,
            gpu_texture: None,
            gpu_view: None,
            normal_gpu_texture: None,
            normal_gpu_view: None,
        })
    }

    // -----------------------------------------------------------------------
    // Embedded fallback
    // -----------------------------------------------------------------------

    fn build_embedded() -> Self {
        debug_log!("TextureAtlas", "build_embedded", "Generating procedural embedded textures (albedo + normal)");

        let mut atlas_img = RgbaImage::new(ATLAS_WIDTH, ATLAS_HEIGHT);
        let mut normal_img = RgbaImage::new(ATLAS_WIDTH, ATLAS_HEIGHT);
        let mut uv_map    = HashMap::new();

        fill_tile_solid(&mut atlas_img, 0, 0, 255, 255, 255);
        fill_tile_flat_normal(&mut normal_img, 0, 0);

        let textures: [(&str, u8, u8, u8); 7] = [
            ("grass_top",    56, 133,  36),
            ("grass_side",   77, 102,  38),
            ("grass_bottom", 102, 66,  26),
            ("dirt",         102, 66,  26),
            ("stone_top",    128, 128, 128),
            ("stone_side",   122, 122, 122),
            ("stone_bottom", 128, 128, 128),
        ];

        for (i, &(name, r, g, b)) in textures.iter().enumerate() {
            let slot = (i + 1) as u32;
            let col  = slot % ATLAS_COLS;
            let row  = slot / ATLAS_COLS;

            fill_tile_noisy(&mut atlas_img, col, row, r, g, b, 20);
            fill_tile_flat_normal_at(&mut normal_img, col, row);

            let u0 = col as f32 / ATLAS_COLS as f32;
            let v0 = row as f32 / ATLAS_ROWS as f32;
            let u1 = (col + 1) as f32 / ATLAS_COLS as f32;
            let v1 = (row + 1) as f32 / ATLAS_ROWS as f32;
            uv_map.insert(name.to_string(), (u0, v0, u1, v1));

            debug_log!(
                "TextureAtlas", "build_embedded",
                "Generated '{}' at slot ({},{})", name, col, row
            );
        }

        debug_log!("TextureAtlas", "build_embedded", "Total procedural textures: {}", uv_map.len());

        let mip_images = Self::generate_mip_chain(&atlas_img);
        let normal_mip_images = Self::generate_mip_chain(&normal_img);
        Self {
            image: atlas_img,
            mip_images,
            normal_image: normal_img,
            normal_mip_images,
            layout: TextureAtlasLayout { uv_map },
            mip_dirty: false,
            gpu_texture: None,
            gpu_view: None,
            normal_gpu_texture: None,
            normal_gpu_view: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Tile-fill helpers
// ---------------------------------------------------------------------------

fn fill_tile_solid(atlas: &mut RgbaImage, col: u32, row: u32, r: u8, g: u8, b: u8) {
    let x0 = col * TILE_SIZE;
    let y0 = row * TILE_SIZE;
    for y in y0..y0+TILE_SIZE { for x in x0..x0+TILE_SIZE {
        atlas.put_pixel(x, y, Rgba([r, g, b, 255]));
    }}
}

fn fill_tile_noisy(atlas: &mut RgbaImage, col: u32, row: u32, r: u8, g: u8, b: u8, noise_strength: u8) {
    let x0 = col * TILE_SIZE;
    let y0 = row * TILE_SIZE;
    for y in y0..y0+TILE_SIZE { for x in x0..x0+TILE_SIZE {
        let h         = simple_hash(x, y);
        let variation = (h as i16 - 128) * noise_strength as i16 / 128;
        let nr = (r as i16 + variation).clamp(0, 255) as u8;
        let ng = (g as i16 + variation).clamp(0, 255) as u8;
        let nb = (b as i16 + variation).clamp(0, 255) as u8;
        atlas.put_pixel(x, y, Rgba([nr, ng, nb, 255]));
    }}
}

/// Fill a tile with a flat normal map (pointing straight up in tangent space).
/// RGB = (128, 128, 255) encodes N = (0, 0, 1) in tangent space.
fn fill_tile_flat_normal(atlas: &mut RgbaImage, col: u32, row: u32) {
    let x0 = col * TILE_SIZE;
    let y0 = row * TILE_SIZE;
    for y in y0..y0+TILE_SIZE { for x in x0..x0+TILE_SIZE {
        atlas.put_pixel(x, y, Rgba([128, 128, 255, 255]));
    }}
}

/// Same as `fill_tile_flat_normal` but takes already-computed x0/y0.
fn fill_tile_flat_normal_at(atlas: &mut RgbaImage, col: u32, row: u32) {
    fill_tile_flat_normal(atlas, col, row);
}

fn simple_hash(x: u32, y: u32) -> u8 {
    let h = x.wrapping_mul(374761393)
        .wrapping_add(y.wrapping_mul(668265263))
        .wrapping_add(1013904223);
    ((h >> 16) & 0xFF) as u8
}