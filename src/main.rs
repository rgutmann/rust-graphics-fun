#[macro_use]
extern crate lazy_static;

extern crate glutin_window;
extern crate graphics;
extern crate opengl_graphics;
extern crate piston;
extern crate core;

use std::f64::consts::PI;
use std::ops::Deref;
use std::sync::RwLock;
use glam::{DVec2};
use glutin_window::GlutinWindow;
use graphics::Context;
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
    static ref SLOW: RwLock<bool> = RwLock::new( false );
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
    /// collider polygon (relative to position as center)
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
    /// Returns intersection point between both bodies or None.
    /// WARNING: Only convex polygons are supported!
    fn intersect(&self, body:&RigidBody) -> Option<DVec2> {
        // see https://stackoverflow.com/questions/753140/how-do-i-determine-if-two-convex-polygons-intersect
        // idea: use slower ray casting algorithm since it can handle non-convex polygons
        // and also returns the intersection point which we might need in the future
        // TODO do both have collider polygons?
        // TODO if no, just check if point is included in the other
        // TODO otherwise iterate over all my and his vertices (to check if they're included in the other)
        // iterate over all my vertices
        let mut intersection_point = None;
        let my_pos = self.pos.clone();
        let other_pos = body.pos.clone();
        // rotate AND translate vertices points
        let my_points:Vec<DVec2> = self.coll.as_ref().unwrap().points.deref().clone()
            .iter().map(|p| { DVec2::from_angle(self.ori).rotate(*p)+my_pos }).collect();
        let other_points:Vec<DVec2> = body.coll.as_ref().unwrap().points.deref().clone()
            .iter().map(|p| { DVec2::from_angle(body.ori).rotate(*p)+other_pos }).collect();
        // first check one side, and if not found from the other side
        intersection_point = Self::is_point_inside(&my_pos, &my_points, &other_points);
        if intersection_point == None {
            intersection_point = Self::is_point_inside(&other_pos, &other_points, &my_points);
        }

        // // default impl for intersection point now is the middle of both centers
        // let intersection = self.pos + (body.pos - self.pos) * DVec2::new(0.5,0.5);
        // Some(intersection)
        intersection_point
    }

    fn is_point_inside(my_pos: &DVec2, my_points: &Vec<DVec2>, other_points: &Vec<DVec2>) -> Option<DVec2> {
        //println!("..my_pos {my_pos:?}, my_points {my_points:?}, other_points {other_points:?}");
        let mut intersection_point = None;

        for p in my_points.iter() {
            // calculate line parameters (slope and offset) from my center to my vertex (any ray to this vertex is sufficient)
            let line1 = Self::calculate_line_params(*my_pos, *p);
            // for the other sides, we need to walk in pairs...
            let other_count = other_points.len();
            let mut intersection_list = vec![];
            // iterate over all other body's sides
            for i in 0..other_count {
                let p1 = other_points[i];
                let mut next_i = i + 1;
                if next_i >= other_count {
                    next_i = 0;
                }
                let p2 = other_points[next_i];
                let line2 = Self::calculate_line_params(p1, p2);
                //println!("..line1 {line1:?}, line2 {line2:?}");
                // is it intersecting or are they parallel? => compare slopes
                if line1.0 != line2.0 {
                    // unequal slope, so there must be an intersection
                    let x = (line2.1 - line1.1) / (line1.0 - line2.0);
                    let y = line1.0 * x + line1.1;
                    //println!("..possible intersection found: {:?}", DVec2::new(x,y));
                    // check if intersection point is within other side limits
                    if x >= p1.x.min(p2.x) && x <= p1.x.max(p2.x) {
                        // check if p is reached before => limits: (my_pos..=p)
                        if x >= my_pos.x.min(p.x) && x <= my_pos.x.max(p.x) {
                            //println!("!!intersection found");
                            intersection_list.push(DVec2::new(x, y));
                        }
                    } else {
                        //println!("..no intersection found");
                    }
                } else {
                    //println!("--parallel lines found");
                }

            } // end of for all other sides

            // check intersection count - only uneven count is a sign of a real intersection
            if (intersection_list.len() % 2) > 0 {
                // found: use last one - we can stop now...
                intersection_point = Some(intersection_list.get(intersection_list.len() - 1).unwrap().clone());
                //println!("!!!!!real intersection found");
                if *(DEBUG.read().unwrap()) { // TODO REMOVE WHEN NOT NEEDED ANYMORE
                    *(PAUSE.write().unwrap()) = true;
                }
                break;
            } else {
                //println!("-----no real intersection found");
            }
            intersection_list.clear();

        } // end of for my vertices
        intersection_point
    }

    /// Calculate slope a1 and offset b1.
    fn calculate_line_params(p1: DVec2, p2: DVec2) -> (f64,f64) {
        let a1 = (p2.y - p1.y) / (p2.x - p1.x);
        let b1 = p2.y - (a1 * p2.x);
        (a1, b1)
    }

    /// Returns coarse boundary (2d-area) of collision polygon after orientation rotation.
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
    /// Resets the previous position (e.g. before a collision happened).
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
    fn body(&self) -> &RigidBody {
        &self.body
    }
    fn collide(&mut self, body:&RigidBody, _coll_pos:&DVec2) {
        self.body.restore_last_pos();
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
                    r2[0].x >= r1[1].x    // r2.left > r1.right
                    || r2[1].x <= r1[0].x // r2.right < r1.left
                    || r2[0].y >= r1[1].y // r2.top > r1.bottom
                    || r2[1].y <= r1[0].y // r2.bottom < r1.top
                );
                if intersect {
                    // finer collision detection for remaining candidates
                    let ibody= (*self.go_list.get(i).unwrap().body()).clone();
                    let jbody= (*self.go_list.get(j).unwrap().body()).clone();
                    if let Some(coll_pos) = ibody.intersect(&jbody) {
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