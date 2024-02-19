extern crate glutin_window;
extern crate graphics;
extern crate opengl_graphics;
extern crate piston;
extern crate core;

use glam::{DVec2};
use glutin_window::GlutinWindow as Window;
use graphics::Context;
use opengl_graphics::{GlGraphics, OpenGL};
use piston::event_loop::{EventSettings, Events};
use piston::input::{RenderArgs, RenderEvent, UpdateArgs, UpdateEvent};
use piston::window::WindowSettings;
use rand::RngCore;
use rand::rngs::OsRng;

const BLACK: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

// Every object that needs to be rendered on screen.
pub trait GameObject {
    fn render(&self, ctxt: &Context, gl: &mut GlGraphics);
    fn update(&mut self, _dt: f64, _ac: &AppContext) {
        // By default do nothing in the update function
    }
}

#[derive(Debug, Default)]
pub struct RigidBody {
    /// position
    pub pos: DVec2,
    /// velocity
    pub vel: DVec2,
    /// acceleration
    pub acc: DVec2,
    /// orientation (rotation)
    pub ori: f64,
    /// torque (angular velocity)
    pub tor: f64,
    /// gravity
    pub grav: f64,
    /// collider polygon
    pub coll: Option<Polygon>,
}
impl RigidBody {
    /// Updates movement and rotation
    pub fn update(&mut self, dt:f64) {
        // move
        self.pos += self.vel * DVec2::new(dt,dt);
        // accelerate
        self.vel += self.acc * DVec2::new(dt,dt);
        if self.grav > 0f64 {
            self.vel += DVec2::new(0f64, self.grav) * DVec2::new(dt,dt);
        }
        // rotate
        self.ori += self.tor * dt;
    }
    /// Updates movement and rotation, but ensures that body stays within boundary and returns a violation vector option
    pub fn update_with_boundary(&mut self, dt:f64, outer:[DVec2;2]) -> Option<DVec2> {
        self.update(dt);
        let mut violation = DVec2::new(0f64, 0f64);
        match &self.coll {
            None => {
                if self.pos.x < outer[0].x { violation[0]+=self.pos.x - outer[0].x; };
                if self.pos.y < outer[0].y { violation[1]+=self.pos.y - outer[0].y; };
                if self.pos.x > outer[1].x { violation[0]+=self.pos.x - outer[1].x; };
                if self.pos.y > outer[1].y { violation[1]+=self.pos.y - outer[1].y; };
            },
            Some(poly) => {
                let mut inner = poly.boundary(dt);
                inner[0] += self.pos;
                inner[1] += self.pos;
                if inner[0].x < outer[0].x { violation.x+=inner[0].x - outer[0].x; };
                if inner[0].y < outer[0].y { violation.y+=inner[0].y - outer[0].y; };
                if inner[1].x > outer[1].x { violation.x+=inner[1].x - outer[1].x; };
                if inner[1].y > outer[1].y { violation.y+=inner[1].y - outer[1].y; };
            }
        }
        self.pos -= violation;
        if violation.x == 0f64 && violation.y == 0f64 { None } else { Some(violation) }
    }
}

#[derive(Debug, Default)]
/// Polygon with is centered / relative to (0,0)
pub struct Polygon {
    points: Box<[DVec2]>,
}
impl Polygon {
    /// Returns a max rectangle shaped boundary area (left upper and right lower point)
    fn boundary(&self, _ori:f64) -> [DVec2;2] {
        let mut result = [ DVec2::ZERO.clone(), DVec2::ZERO.clone() ];
        for p in self.points.iter() {
            if p.x < result[0].x { result[0].x = p.x };
            if p.y < result[0].y { result[0].y = p.y };
            if p.x > result[1].x { result[1].x = p.x };
            if p.y > result[1].y { result[1].y = p.y };
        }
        result
    }
}

#[derive(Debug)]
struct Square {
    body: RigidBody,
    color: [f32; 4],
    size: f64,
}
impl Square {
    pub fn new(body: RigidBody, color:[f32; 4], size:f64) -> Self {
        let half_size = size / 2.0;
        let mut square = Square { body, color, size, };
        square.body.coll = Some(Polygon { points: Box::new([
            DVec2::new( -half_size, -half_size ),
            DVec2::new( half_size, -half_size ),
            DVec2::new( half_size, half_size ),
            DVec2::new( -half_size, half_size ),
            ])
        });
        square
    }
}

impl GameObject for Square {
    fn render(&self, ctxt: &Context, gl: &mut GlGraphics) {
        use graphics::*;
        let square = rectangle::square(0.0, 0.0, self.size);
        let transform = ctxt
            .transform
            .trans(self.body.pos[0], self.body.pos[1])
            .rot_rad(self.body.ori)
            .trans(-(self.size/2.0), -(self.size/2.0));
        // Draw a box rotating around the middle of the screen.
        rectangle(self.color, square, transform, gl);
    }
    fn update(&mut self, dt: f64, ac: &AppContext) {
        let boundary = [ DVec2::ZERO.clone(), DVec2::new(ac.window_size[0], ac.window_size[1]) ];
        let violation_option = self.body.update_with_boundary(dt, boundary);
        // Check boundary violations
        match violation_option {
            None => {},
            Some(violation) => {
                let mut new_vel = self.body.vel;
                // bounce from border violations
                if violation.x < 0f64 { new_vel[0] = - new_vel[0]; };
                if violation.y < 0f64 { new_vel[1] = - new_vel[1]; };
                if violation.x > 0f64 { new_vel[0] = - new_vel[0]; };
                if violation.y > 0f64 { new_vel[1] = - new_vel[1]; };
                // and adapt velocity by rotating +/- 25% of 1 radian (+/- 14 degrees)
                new_vel = DVec2::from_angle(1f64 * random_25perc_var()).rotate(new_vel);
                // reverse rotation and adapt rotation speed by +/- 25%
                self.body.tor = - self.body.tor * (1.0+random_25perc_var());
                self.body.vel = new_vel;
            }
        }
    }
}

pub struct App {
    gl: GlGraphics, // OpenGL drawing backend.
    ac: AppContext,
    go_list: Vec<Box<dyn GameObject>>,
}

pub struct AppContext {
    window_size: [f64; 2],
}

impl App {
    fn render(&mut self, args: &RenderArgs) {
        use graphics::*;

        self.ac.window_size = args.viewport().window_size.clone();

        self.gl.draw(args.viewport(), |c, gl| {
            // Clear the screen.
            clear(BLACK, gl);
            for go in &self.go_list {
                go.render(&c, gl);
            }
        });
    }

    fn update(&mut self, args: &UpdateArgs) {
        for go in self.go_list.iter_mut() {
            go.update(args.dt, &self.ac);
        }
    }
}

// Generates randoms between -25% and +25%
fn random_25perc_var() -> f64 {
    (((OsRng.next_u32()) as f64) / (u32::MAX as f64) / 2.0) - 0.25
}

fn main() {
    // Change this to OpenGL::V2_1 if not working.
    let opengl = OpenGL::V3_2;

    // Create a Glutin window.
    let initial_window_size = [400, 200];
    let mut window: Window = WindowSettings::new("spinning-squares", initial_window_size)
        .graphics_api(opengl)
        .exit_on_esc(true)
        .build()
        .unwrap();

    let center_position = DVec2::new((initial_window_size[0] / 2) as f64, (initial_window_size[1] / 2) as f64);
    let mut go_list :Vec<Box<dyn GameObject>> = vec![];
    for i in 0..=8 {
        let cshard = ((10-i) as f32) / 10.0;
        let max_size = 50.0f64;
        let max_speed = 200.0f64;
        let body = RigidBody {
            ori: 0.0,
            tor: 2.0,
            pos: center_position,
            vel: DVec2::new(max_speed * ((i+2) as f64 / 10.0), max_speed * ((i+2) as f64 / 10.0)),
            grav: 50.0,
            ..Default::default()
        };
        let color = [cshard, cshard, cshard, 1.0];
        let size = max_size * ((10-i) as f64 / 10.0);
        let x = Box::new(Square::new( body, color, size ));
        println!("{:?}", x);
        go_list.push(x);
    }

    // Create a new game and run it.
    let mut app = App {
        gl: GlGraphics::new(opengl),
        ac: AppContext {
            window_size: [(initial_window_size[0]) as f64, (initial_window_size[1]) as f64] as [f64;2],
        },
        go_list
    };

    let mut events = Events::new(EventSettings::new());
    while let Some(e) = events.next(&mut window) {
        if let Some(args) = e.render_args() {
            app.render(&args);
        }
        if let Some(args) = e.update_args() {
            app.update(&args);
        }
    }
}

#[cfg(test)]
mod test {
    use std::ops::{Add, Mul};
    use glam::Vec2;

    #[test]
    fn test_glam() {
        let v1 = Vec2::new(1.0,1.0);
        let v2 = Vec2::new(2.0,3.0);
        let v3 = v1.add(v2);
        println!("{v1}.add({v2})={v3}");

        let v1 = Vec2::new(1.0,1.0);
        let v2 = Vec2::new(2.0,3.0);
        let v3 = v1.mul(v2);
        println!("{v1}.mul({v2})={v3}");
    }
}