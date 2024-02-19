extern crate glutin_window;
extern crate graphics;
extern crate opengl_graphics;
extern crate piston;

use std::ops::{Add, Mul};
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

#[derive(Debug)]
struct Square {
    color: [f32; 4],
    size: f64,
    rotation: f64,
    rotation_speed: f64,
    position: DVec2,
    velocity: DVec2,
}

impl GameObject for Square {
    fn render(&self, ctxt: &Context, gl: &mut GlGraphics) {
        use graphics::*;
        let square = rectangle::square(0.0, 0.0, self.size);
        let transform = ctxt
            .transform
            .trans(self.position[0], self.position[1])
            .rot_rad(self.rotation)
            .trans(-(self.size/2.0), -(self.size/2.0));
        // Draw a box rotating around the middle of the screen.
        rectangle(self.color, square, transform, gl);
    }
    fn update(&mut self, dt: f64, ac: &AppContext) {
        // Rotate 2 radians per second.
        self.rotation += self.rotation_speed * dt;
        // Move into direction
        let mut new_pos = self.position.add(self.velocity.mul(DVec2::new(dt,dt)));
        let mut new_vel = self.velocity;
        // Check boundary violations
        let half_size = self.size / 2.0;
        let mut bounced = false;
        if new_pos[0] < half_size { new_pos[0] = half_size; new_vel[0] = - new_vel[0]; bounced = true; };
        if new_pos[1] < half_size { new_pos[1] = half_size; new_vel[1] = - new_vel[1]; bounced = true; };
        if new_pos[0] > (ac.window_size[0] - half_size) { new_pos[0] = ac.window_size[0] - half_size; new_vel[0] = - new_vel[0]; bounced = true; };
        if new_pos[1] > (ac.window_size[1] - half_size) { new_pos[1] = ac.window_size[1] - half_size; new_vel[1] = - new_vel[1]; bounced = true; };
        if bounced {
            // adapt velocity vector by +/- 25% in x and y direction
            new_vel += new_vel.mul(DVec2::new(random_25perc_var(), random_25perc_var()));
            // adapt rotation speed by +/- 25%
            self.rotation_speed = - self.rotation_speed * (1.0+random_25perc_var());
        }
        self.position = new_pos;
        self.velocity = new_vel;
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
    let mut window: Window = WindowSettings::new("spinning-square", initial_window_size)
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
        let x = Box::new(Square {
            color: [cshard, cshard, cshard, 1.0],
            size: max_size * ((10-i) as f64 / 10.0),
            rotation: 0.0,
            rotation_speed: 2.0,
            position: center_position,
            velocity: DVec2::new(max_speed * ((10-i) as f64 / 10.0), max_speed * ((10-i) as f64 / 10.0)),
        });
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