#[macro_use]
extern crate lazy_static;

extern crate glutin_window;
extern crate graphics;
extern crate opengl_graphics;
extern crate piston;
extern crate core;

use std::f64::consts::PI;
use std::sync::RwLock;
use std::time::{Duration, Instant};
use glam::DVec2;
use glutin_window::GlutinWindow;
use graphics::Context;
use opengl_graphics::{GlGraphics, OpenGL};
use piston::{Button, keyboard, MouseButton, PressEvent, Size, Window};
use piston::event_loop::{Events, EventSettings};
use piston::input::{RenderArgs, RenderEvent, UpdateArgs, UpdateEvent};
use piston::window::WindowSettings;
use rand::RngCore;
use rand::rngs::OsRng;
use rigid_body::{Polygon, RigidBody};
use crate::helper::square_border_at;

mod helper;
mod rigid_body;

const BLACK: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
lazy_static! {
    static ref DEBUG: RwLock<bool> = RwLock::new( false );
    static ref PAUSE: RwLock<bool> = RwLock::new( false );
    static ref SLOW: RwLock<bool> = RwLock::new( false );
}

/// Graphical marker for temporary debug display.
pub struct TempMarker {
    pos: DVec2,
    expiration: Instant,
}
impl TempMarker {
    pub fn new(pos: &DVec2) -> Self {
        TempMarker{  pos: pos.clone(), expiration: Instant::now() + Duration::from_secs(2), }
    }
    pub fn is_expired(&self) -> bool {
        Instant::now() > self.expiration
    }
}

/// Every object that needs to be alive and rendered.
pub trait GameObject {
    /// Render GameObject to view.
    fn render(&self, ctxt: &Context, gl: &mut GlGraphics);
    /// Update GameObject physic state.
    fn update(&mut self, _dt: f64, _ac: &AppContext) {
        // By default do nothing in the update function
    }
    /// Return body for collision detection.
    fn body(&self) -> &RigidBody;
    /// Custom collision behavior.
    fn collide(&mut self, body:&RigidBody, coll_pos:&DVec2);
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
        square.body.mass = (size*size)/100.0;
        square
    }
    fn bounce(&mut self, violation: &DVec2) {
        let mut new_vel = self.body.vel;
        // bounce from border violations
        if violation.x < 0f64 { new_vel.x = f64::abs(new_vel.x); };
        if violation.x > 0f64 { new_vel.x = -f64::abs(new_vel.x); };
        if violation.y < 0f64 { new_vel.y = f64::abs(new_vel.y); };
        if violation.y > 0f64 { new_vel.y = -f64::abs(new_vel.y); };
        // and adapt velocity by rotating +/- 25% of 1 radian (+/- 14 degrees)
        new_vel = DVec2::from_angle(1f64 * helper::random_25perc_var()).rotate(new_vel);
        // reverse rotation and adapt rotation speed by +/- 25%
        self.body.tor = -self.body.tor * (1.0 + helper::random_25perc_var());
        self.body.vel = new_vel;
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
        if *DEBUG.read().unwrap() {
            // velocity vector
            line_from_to(RED, 1.0, self.body.pos, self.body.pos+self.body.vel, ctxt.transform, gl);
            // boundary box
            let boundary = self.body.boundary();
            helper::rectangle_border_from_to(RED, 1.0, boundary[0], boundary[1], ctxt.transform, gl);
        }
    }
    fn update(&mut self, dt: f64, ac: &AppContext) {
        let boundary = [ DVec2::ZERO.clone(), DVec2::new(ac.window_size[0], ac.window_size[1]) ];
        let violation_option = self.body.update_with_boundary(dt, boundary);
        // Check boundary violations
        match violation_option {
            None => {},
            Some(violation) => {
                self.bounce(&violation);
            }
        }
    }
    fn body(&self) -> &RigidBody {
        &self.body
    }
    fn collide(&mut self, body:&RigidBody, _coll_pos:&DVec2) {
        self.body.restore_last_pos(); // just to ensure that the bodies don't remain overlapped

        // TODO real 2d rigid body collision instead of fake direction inversion
        // let vel1 = self.body.vel;
        // let vel2 = body.vel;

        let mut direction = (body.pos-self.body.pos).angle_between(DVec2::new(1.0,0.0));
        if direction < 0.0 { direction+=2.0*PI };
        const Q_PI: f64 = PI / 4.0;
        let violation = if direction < Q_PI || direction >= Q_PI * 7.0  {
            DVec2::new(1.0,0.0)  // right bounce
        } else if direction < Q_PI * 3.0 && direction >= Q_PI * 1.0 {
            DVec2::new(0.0,-1.0) // top bounce
        } else if direction < Q_PI * 5.0 && direction >= Q_PI * 3.0 {
            DVec2::new(-1.0,0.0) // left bounce
        } else if direction < Q_PI * 7.0 && direction >= Q_PI * 5.0 {
            DVec2::new(0.0,1.0)  // bottom bounce
        } else { DVec2::new(0.0,0.0)  /* no bounce */ };
        //println!("coll between pos {:?} and other pos {:?} with angle {:?} PI/4 resulting in {:?}", self.body.pos, body.pos, direction/ Q_PI, &violation);
        self.bounce(&violation);
    }
}

pub struct App {
    gl: GlGraphics, // OpenGL drawing backend.
    ac: AppContext,
    go_list: Vec<Box<dyn GameObject>>,
    marker_list: Vec<TempMarker>,
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
            self.marker_list.retain(|tm|{!tm.is_expired()});
            self.marker_list.iter()
                .for_each(|tm|{ square_border_at(RED,1.0, 4.0, tm.pos, c.transform, gl) });
        });
    }

    fn update(&mut self, args: &UpdateArgs) {
        // movement of game objects
        for go in self.go_list.iter_mut() {
            let mut dt = args.dt;
            if *(SLOW.read().unwrap()) {
                dt = dt / 10.0;
            }
            go.update(dt, &self.ac);
        }
        // coarse collision detection
        let boundaries = self.go_list.iter()
            .map(|go| go.body().boundary())
            .collect::<Vec<[DVec2;2]>>();
        for i in 0..boundaries.len() {
            for j in (i+1)..boundaries.len() {
                // check if i is in collision with j
                let r1 = boundaries[i];
                let r2 = boundaries[j];
                let intersect = !(
                       r1[0].x == r1[1].x && r1[0].y == r1[1].y // r1 has no dimension
                    || r2[0].x == r2[1].x && r2[0].y == r2[1].y // r2 has no dimension
                    || r2[0].x >= r1[1].x // r2.left > r1.right
                    || r2[1].x <= r1[0].x // r2.right < r1.left
                    || r2[0].y >= r1[1].y // r2.top > r1.bottom
                    || r2[1].y <= r1[0].y // r2.bottom < r1.top
                );
                if intersect {
                    // finer collision detection for remaining candidates
                    let ibody= (*self.go_list.get(i).unwrap().body()).clone();
                    let jbody= (*self.go_list.get(j).unwrap().body()).clone();
                    if let Some(coll_pos) = ibody.intersect(&jbody) {
                        if *(DEBUG.read().unwrap()) {
                            // add temporary highlight of position (circle?)
                            self.marker_list.push(TempMarker::new(&coll_pos));
                            //*(PAUSE.write().unwrap()) = true; // TODO REMOVE WHEN NOT NEEDED ANYMORE
                        }
                        // inform about collision
                        //println!("real collision detected between {}::{:?} and {}::{:?}",i,boundaries[i],j,boundaries[j]);
                        self.go_list.get_mut(i).unwrap().collide(&jbody, &coll_pos);
                        self.go_list.get_mut(j).unwrap().collide(&ibody, &coll_pos);
                    } else {
                        //println!("coarse collision ignored between {}::{:?} and {}::{:?}",i,boundaries[i],j,boundaries[j]);
                    }
                }
            }
        }

    }
}

/// Generate initial game objects for startup and restart.
fn generate_game_objects(size: Size) -> Vec<Box<dyn GameObject>> {
    let center_position = DVec2::new(size.width as f64 / 2.0, size.height as f64 / 2.0);
    let mut go_list: Vec<Box<dyn GameObject>> = vec![];
    for i in 0..=8 {
        let cshard = ((i+2) as f32) / 10.0;
        let max_size = 50.0f64;
        let max_speed = 200.0f64;
        let body = RigidBody {
            ori: 0.0,
            tor: 2.0,
            pos: center_position + DVec2::new((i - 4) as f64 * (max_size + 20f64), 0f64),
            vel: DVec2::new(max_speed * ((i + 2) as f64 / 10.0), max_speed * ((i + 2) as f64 / 10.0)),
            grav: 50.0,
            ..Default::default()
        };
        let color = [cshard, cshard, cshard, 1.0];
        let size = max_size * ((10 - i) as f64 / 10.0);
        let x = Box::new(Square::new(body, color, size));
        println!("{:?}", x);
        go_list.push(x);
    }
    go_list
}


fn main() {
    // Change this to OpenGL::V2_1 if not working.
    let opengl = OpenGL::V3_2;

    // Create a Glutin window.
    let initial_window_size = [800, 200];
    let mut window: GlutinWindow = WindowSettings::new("Spinning-squares --- 'P' Pause - 'S' Slow - 'leftMB' Restart - 'rightMB' Debug", initial_window_size)
        .graphics_api(opengl)
        .exit_on_esc(true)
        .build()
        .unwrap();

    let go_list = generate_game_objects(window.size());

    // Create a new game and run it.
    let mut app = App {
        gl: GlGraphics::new(opengl),
        ac: AppContext {
            window_size: [(initial_window_size[0]) as f64, (initial_window_size[1]) as f64] as [f64;2],
        },
        go_list,
        marker_list: vec![],
    };

    let mut events = Events::new(EventSettings::new());
    while let Some(e) = events.next(&mut window) {
        if let Some(args) = e.render_args() {
            app.render(&args);
        }
        if let Some(args) = e.update_args() {
            if !*(PAUSE.read().unwrap()) {
                app.update(&args);
            }
        }
        if let Some(Button::Keyboard(key)) = e.press_args() {
            println!("Pressed key '{:?}'", key);
            if key == keyboard::Key::P {
                let pause = !*(PAUSE.read().unwrap());
                println!("Pause flag switched to '{:?}'", pause);
                *(PAUSE.write().unwrap()) = pause;
            }
            if key == keyboard::Key::S {
                let slow = !*(SLOW.read().unwrap());
                println!("Slow flag switched to '{:?}'", slow);
                *(SLOW.write().unwrap()) = slow;
            }
        }
        if let Some(Button::Mouse(button)) = e.press_args() {
            println!("Pressed mouse button '{:?}'", button);
            if button == MouseButton::Left {
                let mut game_objects = generate_game_objects(window.size());
                app.go_list.clear();
                app.go_list.append(&mut game_objects);
            }
            if button == MouseButton::Right {
                let debug = !*(DEBUG.read().unwrap());
                println!("Debug flag switched to '{:?}'", debug);
                *(DEBUG.write().unwrap()) = debug;
            }
        }
    }
}

#[cfg(test)]
mod test {
    use std::ops::{Add, Mul};
    use glam::Vec2;

    #[test]
    fn test_scrapbook() {
        let v1 = Vec2::new(1.0,1.0);
        let v2 = Vec2::new(2.0,3.0);
        let v3 = v1.add(v2);
        println!("{v1}.add({v2})={v3}");

        let v1 = Vec2::new(1.0,1.0);
        let v2 = Vec2::new(2.0,3.0);
        let v3 = v1.mul(v2);
        println!("{v1}.mul({v2})={v3}");

        let expiation = std::time::Instant::now();
    }

}