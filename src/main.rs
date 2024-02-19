#[macro_use]
extern crate lazy_static;

extern crate glutin_window;
extern crate graphics;
extern crate opengl_graphics;
extern crate piston;
extern crate core;

use std::f64::consts::PI;
use std::sync::RwLock;
use glam::{DVec2};
use glutin_window::GlutinWindow;
use graphics::Context;
use graphics::rectangle::rectangle_by_corners;
use opengl_graphics::{GlGraphics, OpenGL};
use piston::{Button, keyboard, MouseButton, PressEvent, Size, Window};
use piston::event_loop::{EventSettings, Events};
use piston::input::{RenderArgs, RenderEvent, UpdateArgs, UpdateEvent};
use piston::window::WindowSettings;
use rand::RngCore;
use rand::rngs::OsRng;

const BLACK: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
lazy_static! {
    static ref DEBUG: RwLock<bool> = RwLock::new( false );
    static ref PAUSE: RwLock<bool> = RwLock::new( false );
}

// Every object that needs to be rendered on screen.
pub trait GameObject {
    fn render(&self, ctxt: &Context, gl: &mut GlGraphics);
    fn update(&mut self, _dt: f64, _ac: &AppContext) {
        // By default do nothing in the update function
    }
    fn boundary(&self) -> [DVec2;2];
    fn collide(&mut self, body:&RigidBody);
    fn body(&self) -> &RigidBody;
}

#[derive(Debug, Default, Clone)]
pub struct RigidBody {
    /// position
    pub pos: DVec2,
    /// previous position
    pub last_pos: DVec2,
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
        self.last_pos = self.pos.clone();
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
                let mut inner = poly.boundary(self.ori);
                inner[0] += self.pos;
                inner[1] += self.pos;
                if inner[0].x < outer[0].x { violation.x+=inner[0].x - outer[0].x; };
                if inner[0].y < outer[0].y { violation.y+=inner[0].y - outer[0].y; };
                if inner[1].x > outer[1].x { violation.x+=inner[1].x - outer[1].x; };
                if inner[1].y > outer[1].y { violation.y+=inner[1].y - outer[1].y; };
            }
        }
        self.pos -= violation;
        if violation.x == 0f64 && violation.y == 0f64 { None }
        else { Some(violation) }
    }
    fn boundary(&self) -> [DVec2;2] {
        match &self.coll {
            None => { [ self.pos.clone(), self.pos.clone() ] }, // no dimension, just position
            Some(poly) => {
                let mut b = poly.boundary(self.ori);
                b[0]+=self.pos;
                b[1]+=self.pos;
                b
            }
        }
    }
    fn restore_last_pos(&mut self) {
        self.pos = self.last_pos;
    }
}

#[derive(Debug, Default, Clone)]
/// Polygon with is centered / relative to (0,0)
pub struct Polygon {
    points: Box<[DVec2]>,
}
impl Polygon {
    /// Returns a max rectangle shaped boundary area (left upper and right lower point)
    fn boundary(&self, _ori:f64) -> [DVec2;2] {
        // first rotate collider points
        let new_points:Vec<DVec2> = self.points.iter().map(|p| { DVec2::from_angle(_ori).rotate(*p) }).collect();
        // noe determine boundaries
        let mut result = [ DVec2::ZERO.clone(), DVec2::ZERO.clone() ];
        for p in new_points {
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
    fn bounce(&mut self, violation: &DVec2) {
        let mut new_vel = self.body.vel;
        // bounce from border violations
        if violation.x < 0f64 { new_vel.x = f64::abs(new_vel.x); };
        if violation.x > 0f64 { new_vel.x = -f64::abs(new_vel.x); };
        if violation.y < 0f64 { new_vel.y = f64::abs(new_vel.y); };
        if violation.y > 0f64 { new_vel.y = -f64::abs(new_vel.y); };
        // and adapt velocity by rotating +/- 25% of 1 radian (+/- 14 degrees)
        new_vel = DVec2::from_angle(1f64 * random_25perc_var()).rotate(new_vel);
        // reverse rotation and adapt rotation speed by +/- 25%
        self.body.tor = -self.body.tor * (1.0 + random_25perc_var());
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
            let transform = ctxt
                .transform
                .trans(self.body.pos[0], self.body.pos[1]);
            line_from_to(RED, 1.0, DVec2::ZERO.clone(), self.body.vel, transform, gl);
            // boundary box
            let boundary = self.body.boundary();
            let transform = ctxt
                .transform;
            //rectangle_from_to(RED, boundary[0], boundary[1], transform, gl);
            line_from_to(RED, 1.0, boundary[0], DVec2::new(boundary[1].x, boundary[0].y), transform, gl);
            line_from_to(RED, 1.0, DVec2::new(boundary[1].x, boundary[0].y), boundary[1], transform, gl);
            line_from_to(RED, 1.0, boundary[1], DVec2::new(boundary[0].x, boundary[1].y), transform, gl);
            line_from_to(RED, 1.0, DVec2::new(boundary[0].x, boundary[1].y), boundary[0], transform, gl);
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
    fn boundary(&self) -> [DVec2; 2] {
        self.body.boundary()
    }
    fn collide(&mut self, body:&RigidBody) {
        self.body.restore_last_pos();
        let mut direction = (body.pos-self.body.pos).angle_between(DVec2::new(1.0,0.0));
        if direction < 0.0 { direction+=2.0*PI };
        const qPI: f64 = PI / 4.0;
        let violation = if direction < qPI || direction >= qPI * 7.0  {
            DVec2::new(1.0,0.0)  // right bounce
        } else if direction < qPI * 3.0 && direction >= qPI * 1.0 {
            DVec2::new(0.0,-1.0) // top bounce
        } else if direction < qPI * 5.0 && direction >= qPI * 3.0 {
            DVec2::new(-1.0,0.0) // left bounce
        } else if direction < qPI * 7.0 && direction >= qPI * 5.0 {
            DVec2::new(0.0,1.0)  // bottom bounce
        } else { DVec2::new(0.0,0.0)  /* no bounce */ };
        println!("coll between pos {:?} and other pos {:?} with angle {:?} PI/4 resulting in {:?}", self.body.pos, body.pos, direction/qPI, &violation);
        self.bounce(&violation);
    }
    fn body(&self) -> &RigidBody {
        &self.body
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
        // movement of game objects
        for go in self.go_list.iter_mut() {
            go.update(args.dt, &self.ac);
        }
        // coarse collision detection
        let boundaries = self.go_list.iter()
            .map(|go| go.boundary())
            .collect::<Vec<[DVec2;2]>>();
        for i in 0..boundaries.len() {
            for j in (i+1)..boundaries.len() {
                // check if i is in collision with j
                let r1 = boundaries[i];
                let r2 = boundaries[j];
                let intersect = !(
                    r2[0].x >= r1[1].x    // r2.left > r1.right
                    || r2[1].x <= r1[0].x // r2.right < r1.left
                    || r2[0].y >= r1[1].y // r2.top > r1.bottom
                    || r2[1].y <= r1[0].y // r2.bottom < r1.top
                );
                if intersect {
                    println!("collision detected between {}::{:?} and {}::{:?}",i,boundaries[i],j,boundaries[j]);
                    let ibody= (*self.go_list.get(i).unwrap().body()).clone();
                    let jbody= (*self.go_list.get(j).unwrap().body()).clone();
                    self.go_list.get_mut(i).unwrap().collide(&jbody);
                    self.go_list.get_mut(j).unwrap().collide(&ibody);
                }
            }
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
    let initial_window_size = [800, 200];
    let mut window: GlutinWindow = WindowSettings::new("spinning-squares", initial_window_size)
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
        go_list
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

fn generate_game_objects(size: Size) -> Vec<Box<dyn GameObject>> {
    let center_position = DVec2::new(size.width as f64 / 2.0, size.height as f64 / 2.0);
    let mut go_list: Vec<Box<dyn GameObject>> = vec![];
    for i in 0..=8 {
        let cshard = ((10 - i) as f32) / 10.0;
        let max_size = 50.0f64;
        let max_speed = 200.0f64;
        let body = RigidBody {
            ori: 0.0,
            tor: 2.0,
            pos: center_position + DVec2::new((i - 4) as f64 * (max_size + 20f64), 0f64),
            //pos: center_position + DVec2::new((max_size + 20f64), 30f64 + (i-1) as f64 * (max_size + 30f64)),
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
    }
}