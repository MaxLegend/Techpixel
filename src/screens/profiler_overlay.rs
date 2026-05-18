// =============================================================================
// QubePixel — ProfilerOverlay  (detailed pipeline breakdown — right side)
// Shows every rendering stage with CPU timing and world stats.
// F4 to toggle.
// =============================================================================

use egui::{Color32, FontId, Pos2, Align2, Rect};

// ---------------------------------------------------------------------------
// ProfilerFrame — one snapshot per frame, fed from GameScreen::render
// ---------------------------------------------------------------------------
pub struct ProfilerFrame {
    // --- Frame summary -------------------------------------------------------
    pub fps:            f64,
    pub frame_total_us: u128,   // render() wall time (µs)

    // --- Pipeline stages (CPU encoding / staging time) ----------------------
    /// Worker: terrain generation (noise + block placement)
    pub gen_time_us:    u128,
    pub gen_count:      usize,
    /// Worker: mesh building (greedy quads → vertices/indices)
    pub mesh_time_us:   u128,
    pub dirty_count:    usize,
    /// Render: chunk VB/IB GPU upload (mapped_at_creation memcpy)
    pub upload_time_us: u128,
    pub upload_count:   usize,
    /// Render: sky billboard pass (sun/moon quads)
    pub sky_render_us:  u128,
    /// Render: VCT voxel volume upload
    pub vct_upload_us:  u128,
    /// Render: VCT GI compute dispatch
    pub vct_dispatch_us: u128,
    /// Render: main 3D terrain pass (opaque chunks)
    pub render_3d_us:   u128,
    /// Render: transparent chunk pass (glass etc.)
    pub transparent_us: u128,
    /// Render: entity / player skeletal pass
    pub entity_render_us: u128,
    /// Render: block model pass (custom 3D model blocks)
    pub block_model_us: u128,
    /// Render: hand block renderer (held block in first/third person)
    pub hand_block_us:  u128,
    /// Render: water transparent pass
    pub water_render_us: u128,
    /// Render: block selection outline pass
    pub outline_us:     u128,

    // --- World / GPU stats --------------------------------------------------
    pub total_chunks:   usize,
    pub visible_chunks: u32,
    pub culled_chunks:  u32,
    pub lod0_count:     usize,
    pub lod1_count:     usize,
    pub lod2_count:     usize,
    pub worker_pending: usize,
    pub upload_pending: usize,
    pub gpu_queue_depth: usize,
    pub vram_bytes:     u64,

    // --- Chunk render stats -------------------------------------------------
    pub draw_calls:             u32,
    pub shadow_draw_calls:      u32,
    pub visible_triangles:      u32,
    pub visible_vram_bytes:     u64,
    pub upload_bytes:           u64,
    pub upload_throughput_mb_s: f64,

    // --- Camera -------------------------------------------------------------
    pub cam_pos: [f32; 3],

    // --- Main-loop phase timings (from frame_timing globals) ----------------
    /// Time blocked in get_current_texture() — GPU sync point for previous frame.
    pub get_texture_us:    u128,
    /// update() + post_process() time.
    pub update_us:         u128,
    /// egui build_ui() + tessellate + texture uploads.
    pub egui_build_us:     u128,
    /// egui render_draw() command encoding.
    pub egui_render_us:    u128,
    /// queue.submit() + surface.present() (CPU side — near-zero).
    pub submit_present_us: u128,
}

// ---------------------------------------------------------------------------
pub struct ProfilerOverlay {
    pub visible: bool,
    last_frame:  Option<ProfilerFrame>,
}

impl ProfilerOverlay {
    pub fn new() -> Self {
        Self { visible: false, last_frame: None }
    }

    pub fn feed(&mut self, frame: ProfilerFrame) {
        self.last_frame = Some(frame);
    }

    pub fn draw_egui(&self, painter: &egui::Painter, sw: f32) {
        if !self.visible { return; }
        let Some(f) = &self.last_frame else { return };

        let padding = 10.0_f32;
        let line_h  = 18.0_f32;
        let font    = FontId::monospace(12.0);

        let ms_color = |us: u128, warn: u128, crit: u128| -> Color32 {
            if us >= crit { Color32::RED }
            else if us >= warn { Color32::YELLOW }
            else { Color32::GREEN }
        };

        let ms = |us: u128| us as f64 / 1000.0;

        let fps_color = if f.fps >= 60.0 { Color32::GREEN }
            else if f.fps >= 30.0 { Color32::YELLOW }
            else { Color32::RED };

        let frame_wall_ms = if f.fps > 0.0 { 1000.0 / f.fps } else { 0.0 };
        let frame_cpu_ms  = ms(f.frame_total_us);
        let _gpu_gap_ms    = (frame_wall_ms - frame_cpu_ms).max(0.0);
        let frame_color   = if frame_wall_ms > 33.3 { Color32::RED }
            else if frame_wall_ms > 16.6 { Color32::YELLOW }
            else { Color32::GREEN };

        let avg_tris = if f.draw_calls > 0 { f.visible_triangles / f.draw_calls } else { 0 };
        let avg_vram_kb = if f.draw_calls > 0 {
            f.visible_vram_bytes as f64 / f.draw_calls as f64 / 1024.0
        } else { 0.0 };

        let upload_kb = f.upload_bytes as f64 / 1024.0;

        // Compute total of all tracked CPU stages
        let tracked_total_us = f.gen_time_us + f.mesh_time_us + f.upload_time_us
            + f.sky_render_us + f.vct_upload_us + f.vct_dispatch_us
            + f.render_3d_us + f.transparent_us + f.entity_render_us
            + f.block_model_us + f.hand_block_us + f.water_render_us + f.outline_us;

        let mut lines: Vec<(String, Color32)> = vec![
            // Header
            ("=== PROFILER ===".to_owned(),
                Color32::from_rgb(255, 180, 60)),

            // Frame summary
            (format!("FPS:  {:.1}  wall={:.2}ms  CPU={:.2}ms",
                f.fps, frame_wall_ms, frame_cpu_ms),
                fps_color),
            (format!("3D encode: {:.2} ms  tracked: {:.2} ms",
                frame_cpu_ms, ms(tracked_total_us)),
                frame_color),

            // Separator
            ("--- CPU encode pipeline ---".to_owned(),
                Color32::from_rgb(80, 80, 80)),

            // Terrain + mesh
            (format!(" 1. Gen:       {:6.2} ms  ({} chunks)",
                ms(f.gen_time_us), f.gen_count),
                ms_color(f.gen_time_us, 5_000, 15_000)),
            (format!(" 2. Mesh:      {:6.2} ms  ({} dirty)",
                ms(f.mesh_time_us), f.dirty_count),
                ms_color(f.mesh_time_us, 5_000, 15_000)),
            (format!(" 3. Ch.Upload: {:6.2} ms  {} bufs  {:.0}KB",
                ms(f.upload_time_us), f.upload_count, upload_kb),
                ms_color(f.upload_time_us, 2_000, 5_000)),
            // Sky
            (format!(" 4. Sky:       {:6.2} ms",
                ms(f.sky_render_us)),
                ms_color(f.sky_render_us, 1_000, 3_000)),
            // VCT
            (format!(" 5. VCT Upld:  {:6.2} ms",
                ms(f.vct_upload_us)),
                ms_color(f.vct_upload_us, 1_000, 3_000)),
            (format!(" 6. VCT GI:    {:6.2} ms",
                ms(f.vct_dispatch_us)),
                ms_color(f.vct_dispatch_us, 1_000, 4_000)),
            // Render passes
            (format!(" 7. 3D Render: {:6.2} ms",
                ms(f.render_3d_us)),
                ms_color(f.render_3d_us, 2_000, 5_000)),
            (format!(" 8. Transpnt:  {:6.2} ms",
                ms(f.transparent_us)),
                ms_color(f.transparent_us, 500, 2_000)),
            (format!(" 9. Entity:    {:6.2} ms",
                ms(f.entity_render_us)),
                ms_color(f.entity_render_us, 500, 2_000)),
            (format!("10. BlkModel:  {:6.2} ms",
                ms(f.block_model_us)),
                ms_color(f.block_model_us, 500, 2_000)),
            (format!("11. Hand:      {:6.2} ms",
                ms(f.hand_block_us)),
                ms_color(f.hand_block_us, 200, 1_000)),
            (format!("12. Water:     {:6.2} ms",
                ms(f.water_render_us)),
                ms_color(f.water_render_us, 500, 2_000)),
            (format!("13. Outline:   {:6.2} ms",
                ms(f.outline_us)),
                ms_color(f.outline_us, 500, 2_000)),

            // Separator
            ("--- frame phases (wall breakdown) ---".to_owned(),
                Color32::from_rgb(80, 80, 80)),

            // get_current_texture: this IS "where the GPU time goes" from the CPU's view.
            // The CPU blocks here waiting for the swapchain to release the next image,
            // which happens after the GPU finishes executing the PREVIOUS frame.
            (format!("A. GPU sync (get_tex): {:6.2} ms  ← GPU prev frame",
                ms(f.get_texture_us)),
                if f.get_texture_us > 10_000 { Color32::RED }
                else if f.get_texture_us > 5_000 { Color32::YELLOW }
                else { Color32::from_rgb(255, 220, 120) }),
            (format!("B. Update:             {:6.2} ms",
                ms(f.update_us)),
                ms_color(f.update_us, 2_000, 5_000)),
            (format!("C. egui build+tess:    {:6.2} ms",
                ms(f.egui_build_us)),
                ms_color(f.egui_build_us, 2_000, 5_000)),
            (format!("D. 3D encode (above):  {:6.2} ms",
                ms(f.frame_total_us)),
                ms_color(f.frame_total_us, 2_000, 8_000)),
            (format!("E. egui render enc:    {:6.2} ms",
                ms(f.egui_render_us)),
                ms_color(f.egui_render_us, 1_000, 3_000)),
            (format!("F. submit+present:     {:6.2} ms",
                ms(f.submit_present_us)),
                ms_color(f.submit_present_us, 500, 2_000)),
            {
                let accounted = f.get_texture_us + f.update_us + f.egui_build_us
                    + f.frame_total_us + f.egui_render_us + f.submit_present_us;
                let wall = if f.fps > 0.0 { (1_000_000.0 / f.fps) as u128 } else { 0 };
                let unaccounted = (wall as i128 - accounted as i128).max(0) as u128;
                (format!("Σ accounted: {:.2} ms / wall {:.2} ms  (gap {:.2} ms)",
                    ms(accounted), ms(wall), ms(unaccounted)),
                    Color32::from_rgb(160, 160, 160))
            },

            // Separator
            ("--- chunk draw ---".to_owned(),
                Color32::from_rgb(80, 80, 80)),

            (format!("Draws: {}  shadow: {}  tris: {}",
                f.draw_calls, f.shadow_draw_calls, f.visible_triangles),
                Color32::from_rgb(180, 220, 255)),
            (format!("Avg/chunk: {} tris  {:.1} KB VRAM",
                avg_tris, avg_vram_kb),
                Color32::from_rgb(160, 200, 255)),
            (format!("Staging: {:.0} KB/frame  {:.0} MB/s",
                upload_kb, f.upload_throughput_mb_s),
                ms_color(f.upload_bytes as u128 * 1000, 512_000, 2_048_000)),
            (format!("Vis.VRAM: {:.2} MB / {:.2} MB total",
                f.visible_vram_bytes as f64 / (1024.0*1024.0),
                f.vram_bytes as f64 / (1024.0*1024.0)),
                Color32::from_rgb(0, 230, 230)),

            // Separator
            ("--- world ---".to_owned(),
                Color32::from_rgb(80, 80, 80)),

            (format!("Chunks: {}  vis={}  cull={}",
                f.total_chunks, f.visible_chunks, f.culled_chunks),
                Color32::from_rgb(180, 220, 255)),
            (format!("LOD: 0={}  1={}  2={}",
                f.lod0_count, f.lod1_count, f.lod2_count),
                Color32::from_rgb(200, 180, 255)),
            (format!("Worker pending: {}   upload Q: {}",
                f.worker_pending, f.gpu_queue_depth),
                if f.gpu_queue_depth > 64 { Color32::YELLOW }
                else { Color32::from_rgb(180, 180, 180) }),

            // Separator
            ("--- camera ---".to_owned(),
                Color32::from_rgb(80, 80, 80)),

            (format!("Pos: ({:.1}, {:.1}, {:.1})",
                f.cam_pos[0], f.cam_pos[1], f.cam_pos[2]),
                Color32::from_rgb(220, 220, 180)),
        ];

        // Annotate the heaviest CPU stage (indices 5..17 = stages 1-13)
        let stages = [
            (f.gen_time_us,          5usize),
            (f.mesh_time_us,         6),
            (f.upload_time_us,       7),
            (f.sky_render_us,        8),
            (f.vct_upload_us,        9),
            (f.vct_dispatch_us,      10),
            (f.render_3d_us,         11),
            (f.transparent_us,       12),
            (f.entity_render_us,     13),
            (f.block_model_us,       14),
            (f.hand_block_us,        15),
            (f.water_render_us,      16),
            (f.outline_us,           17),
        ];
        if let Some(&(worst_us, worst_idx)) = stages.iter().max_by_key(|&&(us, _)| us) {
            if worst_us > 500 {
                let entry = &mut lines[worst_idx];
                entry.0 = format!("{} ◄", entry.0);
                entry.1 = Color32::from_rgb(255, 120, 120);
            }
        }

        // ---- layout --------------------------------------------------------
        let bg_w   = 320.0_f32;
        let bg_h   = lines.len() as f32 * line_h + padding;
        let x_text = sw - padding - bg_w + 8.0;
        let y0     = padding;

        painter.rect_filled(
            Rect::from_min_size(
                Pos2::new(x_text - 8.0, y0 - 6.0),
                egui::Vec2::new(bg_w, bg_h),
            ),
            4.0,
            Color32::from_rgba_unmultiplied(0, 0, 0, 150),
        );

        let mut y = y0;
        for (text, color) in &lines {
            painter.text(
                Pos2::new(x_text, y),
                Align2::LEFT_TOP,
                text.as_str(),
                font.clone(),
                *color,
            );
            y += line_h;
        }
    }
}
