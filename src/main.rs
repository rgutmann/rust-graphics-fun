extern crate glutin_window;
extern crate graphics;
extern crate opengl_graphics;
extern crate piston;

use glutin_window::GlutinWindow as Window;
use graphics::math::{Vec2d};
use opengl_graphics::{GlGraphics, OpenGL};
use piston::event_loop::{EventSettings, Events};
use piston::input::{RenderArgs, RenderEvent, UpdateArgs, UpdateEvent};
use piston::window::WindowSettings;
use rand::RngCore;
use rand::rngs::OsRng;

pub struct App {
    gl: GlGraphics, // OpenGL drawing backend.
    square_size: f64,
    square_rotation: f64,
    square_position: Vec2d,
    square_velocity: Vec2d, // direction + speed
    render_window_size: [f64; 2],
}

impl App {
    fn render(&mut self, args: &RenderArgs) {
        use graphics::*;

        const GREEN: [f32; 4] = [0.0, 1.0, 0.0, 1.0];
        const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];

        let square = rectangle::square(0.0, 0.0, self.square_size);
        //let (win_x_size, win_y_size) = (args.window_size[0] / 2.0, args.window_size[1] / 2.0);

        self.gl.draw(args.viewport(), |c, gl| {
            // Clear the screen.
            clear(GREEN, gl);

            let transform = c
                .transform
                .trans(self.square_position[0], self.square_position[1])
                .rot_rad(self.square_rotation)
                .trans(-(self.square_size/2.0), -(self.square_size/2.0));

            // Draw a box rotating around the middle of the screen.
            rectangle(RED, square, transform, gl);
        });

        // update render_window_size in case window was resized
        self.render_window_size = args.window_size;
    }

    fn update(&mut self, args: &UpdateArgs) {
        // Rotate 2 radians per second.
        self.square_rotation += 2.0 * args.dt;
        // Move into direction
        let mut new_pos = self.square_position;
        let mut new_vel = self.square_velocity;
        // let old_pos = new_pos.clone();
        // let old_vel = new_vel.clone();
        new_pos[0] = new_pos[0] + (self.square_velocity[0] * args.dt);
        new_pos[1] = new_pos[1] + (self.square_velocity[1] * args.dt);
        //println!("translation {:?}::{:?} -> {:?}::{:?}", old_pos, old_vel, new_pos, new_vel);
        // Check boundary violations
        let half_size = self.square_size / 2.0;
        let mut bounced = false;
        if new_pos[0] < half_size { new_pos[0] = half_size; new_vel[0] = - new_vel[0]; bounced = true; };
        if new_pos[1] < half_size { new_pos[1] = half_size; new_vel[1] = - new_vel[1]; bounced = true; };
        if new_pos[0] > (self.render_window_size[0] - half_size) { new_pos[0] = self.render_window_size[0] - half_size; new_vel[0] = - new_vel[0]; bounced = true; };
        if new_pos[1] > (self.render_window_size[1] - half_size) { new_pos[1] = self.render_window_size[1] - half_size; new_vel[1] = - new_vel[1]; bounced = true; };
        if bounced {
            // adapt velocity vector by +/- 25%
            let random_x = (((OsRng.next_u32()) as f64) / (u32::MAX as f64) / 2.0) - 0.25;
            let random_y = (((OsRng.next_u32()) as f64) / (u32::MAX as f64) / 2.0) - 0.25;
            new_vel[0] = new_vel[0] + (new_vel[0]*random_x);
            new_vel[1] = new_vel[1] + (new_vel[1]*random_y);
        }
        self.square_position = new_pos;
        self.square_velocity = new_vel;
        //println!("   boundary {:?}::{:?} -> {:?}::{:?}", old_pos, old_vel, new_pos, new_vel);
    }
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

    // Create a new game and run it.
    let mut app = App {
        gl: GlGraphics::new(opengl),
        square_size: 50.0,
        square_rotation: 0.0,
        square_position: [ (initial_window_size[0] / 2) as f64, (initial_window_size[1] / 2) as f64 ],
        square_velocity: [ 200.0, 200.0 ],
        render_window_size: [ initial_window_size[0] as f64, initial_window_size[1] as f64 ],
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

