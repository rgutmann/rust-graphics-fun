use graphics::types::Radius;
use glam::DVec2;
use graphics::math::Matrix2d;
use opengl_graphics::GlGraphics;
use crate::{OsRng, RngCore};

/// Opengl helper function for drawing a simple rectangle.
pub fn rectangle_border_from_to(color: [f32; 4], radius: Radius, left_upper: DVec2, right_lower: DVec2, transform: Matrix2d, gl: &mut GlGraphics) {
    use graphics::*;
    line_from_to(color, radius, left_upper, DVec2::new(right_lower.x, left_upper.y), transform, gl);
    line_from_to(color, radius, DVec2::new(right_lower.x, left_upper.y), right_lower, transform, gl);
    line_from_to(color, radius, right_lower, DVec2::new(left_upper.x, right_lower.y), transform, gl);
    line_from_to(color, radius, DVec2::new(left_upper.x, right_lower.y), left_upper, transform, gl);
}

/// Generates randoms between -25% and +25%.
pub fn random_25perc_var() -> f64 {
    (((OsRng.next_u32()) as f64) / (u32::MAX as f64) / 2.0) - 0.25
}
