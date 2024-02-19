extern crate glutin_window;
extern crate graphics;
extern crate opengl_graphics;
extern crate piston;

use std::ops::{Add, Mul};
use glam::{DVec2};
use glutin_window::GlutinWindow as Window;
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
    square_rotation_speed: f64,
    square_position: DVec2,
    square_velocity: DVec2, // direction + speed
    render_window_size: [f64;2],
}

impl App {
    fn render(&mut self, args: &RenderArgs) {
        use graphics::*;

        const GREEN: [f32; 4] = [0.0, 1.0, 0.0, 1.0];
        const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];

        let square = rectangle::square(0.0, 0.0, self.square_size);

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
        self.square_rotation += self.square_rotation_speed * args.dt;
        // Move into direction
        let mut new_pos = self.square_position.add(self.square_velocity.mul(DVec2::new(args.dt,args.dt)));
        let mut new_vel = self.square_velocity;
        // Check boundary violations
        let half_size = self.square_size / 2.0;
        let mut bounced = false;
        if new_pos[0] < half_size { new_pos[0] = half_size; new_vel[0] = - new_vel[0]; bounced = true; };
        if new_pos[1] < half_size { new_pos[1] = half_size; new_vel[1] = - new_vel[1]; bounced = true; };
        if new_pos[0] > (self.render_window_size[0] - half_size) { new_pos[0] = self.render_window_size[0] - half_size; new_vel[0] = - new_vel[0]; bounced = true; };
        if new_pos[1] > (self.render_window_size[1] - half_size) { new_pos[1] = self.render_window_size[1] - half_size; new_vel[1] = - new_vel[1]; bounced = true; };
        if bounced {
            // adapt velocity vector by +/- 25% in x and y direction
            let random_x = (((OsRng.next_u32()) as f64) / (u32::MAX as f64) / 2.0) - 0.25;
            let random_y = (((OsRng.next_u32()) as f64) / (u32::MAX as f64) / 2.0) - 0.25;
            new_vel=new_vel.add(new_vel.mul(DVec2::new(random_x,random_y)));
            // adapt rotation speed by +/- 25%
            let random_r = (((OsRng.next_u32()) as f64) / (u32::MAX as f64) / 2.0) - 0.25;
            self.square_rotation_speed = - (self.square_rotation_speed + self.square_rotation_speed * random_r);
        }
        self.square_position = new_pos;
        self.square_velocity = new_vel;
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
        square_rotation_speed: 2.0,
        square_position: DVec2::new((initial_window_size[0] / 2) as f64, (initial_window_size[1] / 2) as f64 ),
        square_velocity: DVec2::new(200.0, 200.0),
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