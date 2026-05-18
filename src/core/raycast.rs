// =============================================================================
// QubePixel — DDA Voxel Raycast (Amanatides & Woo, 1987)
// =============================================================================
//
// Casts a ray from a point through the voxel grid and returns the first
// solid block hit, along with the face normal through which the ray entered.
//
// Used by: block highlight (Phase 2), block breaking (Phase 4),
//          block placement (Phase 5).
// =============================================================================

use glam::{IVec3, Vec3};
use crate::{ext_debug_log, flow_debug_log};

/// Maximum distance (in world units) a raycast will travel.
pub const MAX_RAYCAST_DISTANCE: f32 = 8.0;

/// Hard cap on raycast iterations to prevent infinite loops.
const MAX_RAYCAST_STEPS: u32 = 256;

/// Result of a voxel raycast — the block the player is looking at.
#[derive(Debug, Clone, Copy)]
pub struct RaycastResult {
    /// World-space voxel position of the hit block.
    pub block_pos: IVec3,
    /// Normal of the face through which the ray entered the block.
    /// Points outward (towards the ray origin).
    /// Used for block placement: new_block_pos = block_pos + normal.
    pub normal: IVec3,
    /// Block type ID at the hit position.
    pub block_id: u8,
}

/// DDA voxel traversal (Amanatides & Woo, 1987).
///
/// Walks the ray voxel-by-voxel, checking `get_block` at each step.
/// Returns the **first solid block** (block_id != 0) encountered within
/// `max_distance` of the origin, together with the entry face normal.
///
/// # Parameters
/// - `origin` — ray start (world-space, typically camera position)
/// - `direction` — ray direction (will be normalised internally)
/// - `max_distance` — maximum travel distance in world units
/// - `get_block` — closure `(wx, wy, wz) -> u8`; returns 0 for air
pub fn dda_raycast(
    origin: Vec3,
    direction: Vec3,
    max_distance: f32,
    mut get_block: impl FnMut(i32, i32, i32) -> u8,
) -> Option<RaycastResult> {
    if direction.length_squared() < 1e-10 {
        return None;
    }
    let dir = direction.normalize();

    // Nudge origin slightly along the ray direction to avoid
    // floating-point edge cases when the origin lies exactly on
    // a voxel boundary (causes DDA to step into the wrong voxel).
    let origin = origin + dir * 0.001;

    // Current voxel (the one containing the origin).
    let mut ix = origin.x.floor() as i32;
    let mut iy = origin.y.floor() as i32;
    let mut iz = origin.z.floor() as i32;

    // Step direction along each axis (+1 or -1).
    let step_x: i32 = if dir.x >= 0.0 { 1 } else { -1 };
    let step_y: i32 = if dir.y >= 0.0 { 1 } else { -1 };
    let step_z: i32 = if dir.z >= 0.0 { 1 } else { -1 };

    // How far along the ray we must travel to cross one full voxel on each axis.
    let t_delta_x = if dir.x.abs() > 1e-10 { 1.0 / dir.x.abs() } else { f32::INFINITY };
    let t_delta_y = if dir.y.abs() > 1e-10 { 1.0 / dir.y.abs() } else { f32::INFINITY };
    let t_delta_z = if dir.z.abs() > 1e-10 { 1.0 / dir.z.abs() } else { f32::INFINITY };

    // Parametric distance to the *first* voxel boundary on each axis.
    let mut t_max_x = if dir.x > 0.0 {
        (ix as f32 + 1.0 - origin.x) * t_delta_x
    } else if dir.x < 0.0 {
        (origin.x - ix as f32) * t_delta_x
    } else {
        f32::INFINITY
    };
    let mut t_max_y = if dir.y > 0.0 {
        (iy as f32 + 1.0 - origin.y) * t_delta_y
    } else if dir.y < 0.0 {
        (origin.y - iy as f32) * t_delta_y
    } else {
        f32::INFINITY
    };
    let mut t_max_z = if dir.z > 0.0 {
        (iz as f32 + 1.0 - origin.z) * t_delta_z
    } else if dir.z < 0.0 {
        (origin.z - iz as f32) * t_delta_z
    } else {
        f32::INFINITY
    };

    for _step in 0..MAX_RAYCAST_STEPS {
        // Determine which axis boundary is closest and step through it.
        let face = if t_max_x < t_max_y {
            if t_max_x < t_max_z {
                if t_max_x > max_distance { break; }
                ix += step_x;
                t_max_x += t_delta_x;
                IVec3::new(-step_x, 0, 0)
            } else {
                if t_max_z > max_distance { break; }
                iz += step_z;
                t_max_z += t_delta_z;
                IVec3::new(0, 0, -step_z)
            }
        } else {
            if t_max_y < t_max_z {
                if t_max_y > max_distance { break; }
                iy += step_y;
                t_max_y += t_delta_y;
                IVec3::new(0, -step_y, 0)
            } else {
                if t_max_z > max_distance { break; }
                iz += step_z;
                t_max_z += t_delta_z;
                IVec3::new(0, 0, -step_z)
            }
        };

        // Check the voxel we just stepped into.
        let block_id = get_block(ix, iy, iz);
        if block_id != 0 {
            ext_debug_log!(
                "Raycast", "dda_raycast",
                "Hit block_id={} at ({}, {}, {}) normal=({}, {}, {})",
                block_id, ix, iy, iz, face.x, face.y, face.z
            );
            return Some(RaycastResult {
                block_pos: IVec3::new(ix, iy, iz),
                normal:    face,
                block_id,
            });
        }
    }

    flow_debug_log!("Raycast", "dda_raycast", "No block hit within distance");
    None
}

// ---------------------------------------------------------------------------
// Ray–AABB intersection
// ---------------------------------------------------------------------------

/// Test whether a ray intersects an axis-aligned bounding box.
///
/// Returns `Some(t)` where *t* is the parametric distance along the ray to
/// the **entry** point of the box, or `None` if the ray misses.
///
/// The AABB is defined by its `min` and `max` corners (world-space).
/// The ray is `origin + t * direction` with `t >= 0`.
pub fn ray_aabb_intersection(
    ray_origin: Vec3,
    ray_dir:    Vec3,
    aabb_min:   Vec3,
    aabb_max:   Vec3,
) -> Option<f32> {
    // Slab method: for each axis compute entry/exit t values.
    let mut t_enter = 0.0f32;
    let mut t_exit  = f32::MAX;

    for i in 0..3 {
        let o = ray_origin[i];
        let d = ray_dir[i];
        let mn = aabb_min[i];
        let mx = aabb_max[i];

        if d.abs() < 1e-10 {
            // Ray parallel to slab — origin must be inside
            if o < mn || o > mx {
                return None;
            }
        } else {
            let inv_d = 1.0 / d;
            let t1 = (mn - o) * inv_d;
            let t2 = (mx - o) * inv_d;
            let (t_near, t_far) = if t1 < t2 { (t1, t2) } else { (t2, t1) };

            t_enter = t_enter.max(t_near);
            t_exit  = t_exit.min(t_far);

            if t_enter > t_exit {
                return None;
            }
        }
    }

    if t_exit < 0.0 {
        return None;
    }

    Some(t_enter.max(0.0))
}

// ---------------------------------------------------------------------------
// Möller–Trumbore ray–triangle intersection
// ---------------------------------------------------------------------------

/// Returns `true` if the ray `origin + t * dir` (t > 0) intersects the
/// triangle (`v0`, `v1`, `v2`).  Backface hits are accepted (two-sided).
pub fn ray_triangle_intersect(origin: Vec3, dir: Vec3, v0: Vec3, v1: Vec3, v2: Vec3) -> bool {
    const EPSILON: f32 = 1e-7;
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let h = dir.cross(edge2);
    let a = edge1.dot(h);
    if a.abs() < EPSILON { return false; } // ray parallel to triangle
    let f = 1.0 / a;
    let s = origin - v0;
    let u = f * s.dot(h);
    if u < 0.0 || u > 1.0 { return false; }
    let q = s.cross(edge1);
    let v = f * dir.dot(q);
    if v < 0.0 || u + v > 1.0 { return false; }
    let t = f * edge2.dot(q);
    t > EPSILON
}