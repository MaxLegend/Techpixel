// =============================================================================
// QubePixel — UploadWorker  (parallel CPU staging + main-thread GPU upload)
// =============================================================================

use std::sync::mpsc;
use std::thread;
use std::time::Instant;
use crate::{debug_log, flow_debug_log};
use crate::screens::game_3d_pipeline::Vertex3D;

/// Distinguishes which GPU mesh store an upload job belongs to.
///
/// Opaque  — main chunk pipeline (depth-write, no blend).
/// Water   — fluid pipeline (alpha blend, animated texture).
/// Glass   — transparent solid pipeline (alpha blend, pbr_vct shader).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshKind {
    Opaque,
    Water,
    Glass,
}

// ---------------------------------------------------------------------------
// UploadJob — raw mesh data sent from render thread to worker
// ---------------------------------------------------------------------------
pub struct UploadJob {
    pub key:       (i32, i32, i32),
    pub vertices:  Vec<Vertex3D>,
    pub indices:   Vec<u32>,
    pub aabb_min:  [f32; 3],
    pub aabb_max:  [f32; 3],
    /// Which pipeline / storage map this mesh feeds into.
    pub kind:      MeshKind,
}

// ---------------------------------------------------------------------------
// UploadPacked — CPU-packed mesh data ready for GPU buffer creation
// ---------------------------------------------------------------------------
pub struct UploadPacked {
    pub key:          (i32, i32, i32),
    pub vertex_data:  Vec<u8>,
    pub index_data:   Vec<u8>,
    pub index_count:  u32,
    pub aabb_min:     [f32; 3],
    pub aabb_max:     [f32; 3],
    pub kind:         MeshKind,
}

// ---------------------------------------------------------------------------
// UploadWorker
// ---------------------------------------------------------------------------
pub struct UploadWorker {
    /// Wrapped in Option so it can be explicitly dropped before join(),
    /// causing the background thread's recv() to return Err and exit cleanly.
    job_tx:    Option<mpsc::Sender<UploadJob>>,
    packed_rx: mpsc::Receiver<UploadPacked>,
    thread:    Option<thread::JoinHandle<()>>,
    pending:   usize,
}

impl UploadWorker {
    /// Spawn the background packing thread.  No GPU resources needed.
    pub fn new() -> Self {
        let (job_tx, job_rx)       = mpsc::channel::<UploadJob>();
        let (packed_tx, packed_rx) = mpsc::channel::<UploadPacked>();

        let handle = thread::Builder::new()
            .name("gpu-staging".into())
            .spawn(move || {
                // No staging buffer needed: bytemuck::cast_slice is a zero-cost
                // reinterpret of the existing Vec<Vertex3D> / Vec<u32> memory.
                // One allocation per job (the to_vec() copy) is unavoidable because
                // data must cross the channel boundary, but the old 4–36 MB staging
                // buffer that doubled peak RAM usage is eliminated.
                let mut total_packed = 0usize;

                loop {
                    let first = match job_rx.recv() {
                        Ok(job) => job,
                        Err(_)  => break,
                    };

                    let mut batch = Vec::with_capacity(64);
                    batch.push(first);
                    while let Ok(job) = job_rx.try_recv() {
                        batch.push(job);
                    }

                    let t0        = Instant::now();
                    let batch_len = batch.len();
                    let mut packed = 0usize;

                    for job in batch {
                        if job.vertices.is_empty() || job.indices.is_empty() {
                            continue;
                        }

                        // Zero-copy reinterpret + one owned copy for the channel.
                        let vertex_data: Vec<u8> =
                            bytemuck::cast_slice::<Vertex3D, u8>(&job.vertices).to_vec();
                        let index_data: Vec<u8> =
                            bytemuck::cast_slice::<u32, u8>(&job.indices).to_vec();

                        packed += 1;
                        total_packed += 1;

                        let _ = packed_tx.send(UploadPacked {
                            key:         job.key,
                            vertex_data,
                            index_data,
                            index_count: job.indices.len() as u32,
                            aabb_min:    job.aabb_min,
                            aabb_max:    job.aabb_max,
                            kind:        job.kind,
                        });
                    }

                    let us = t0.elapsed().as_micros();
                    flow_debug_log!(
                        "UploadWorker", "thread",
                        "[PERF] batch={} packed={} total={} time={:.2}ms",
                        batch_len, packed, total_packed,
                        us as f64 / 1000.0,
                    );
                }

                debug_log!("UploadWorker", "thread", "Background thread exiting cleanly");
            })
            .expect("Failed to spawn gpu-staging thread");

        debug_log!("UploadWorker", "new", "Spawned staging worker thread");

        Self {
            job_tx: Some(job_tx),
            packed_rx,
            thread: Some(handle),
            pending: 0,
        }
    }

    // -------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------

    /// Submit a batch of mesh jobs for async CPU packing.  Non-blocking.
    pub fn submit(&mut self, jobs: Vec<UploadJob>) {
        self.pending += jobs.len();
        if let Some(tx) = &self.job_tx {
            for job in jobs {
                let _ = tx.send(job);
            }
        }
    }

    /// Non-blocking poll for packed results.  Call once per frame.
    pub fn poll(&mut self) -> Vec<UploadPacked> {
        let mut results = Vec::new();
        while let Ok(r) = self.packed_rx.try_recv() {
            self.pending = self.pending.saturating_sub(1);
            results.push(r);
        }
        results
    }

    pub fn pending_count(&self) -> usize { self.pending }

    #[allow(dead_code)]
    pub fn is_busy(&self) -> bool { self.pending > 0 }
}

impl Drop for UploadWorker {
    fn drop(&mut self) {
        debug_log!("UploadWorker", "drop", "Shutting down staging worker");

        // IMPORTANT: drop the sender BEFORE joining the thread.
        // This closes the channel, causing job_rx.recv() in the background
        // thread to return Err(_), which breaks the loop and lets join() return.
        // Without this, join() would deadlock because the sender (job_tx)
        // would still be alive during the join, keeping the channel open.
        drop(self.job_tx.take());

        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
            debug_log!("UploadWorker", "drop", "Staging thread joined successfully");
        }
    }
}