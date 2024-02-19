use glam::DVec2;
use std::ops::Deref;
use crate::DEBUG;

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
    /// mass
    pub mass: f64,
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
    pub fn intersect(&self, body:&RigidBody) -> Option<DVec2> {
        // see https://stackoverflow.com/questions/753140/how-do-i-determine-if-two-convex-polygons-intersect
        // idea: use slower ray casting algorithm since it can handle non-convex polygons
        // and also returns the intersection point which we might need in the future

        // do both have collider polygons? (this fn actually shouldn't be called for that)
        if self.coll == None || body.coll == None {
            return None; // one has no dimension -> nothing to collide, nothing to check
        }

        // iterate over all my and his vertices (to check if they're included in the other)
        let my_pos = self.pos.clone();
        let other_pos = body.pos.clone();
        // rotate AND translate vertices points
        let my_points:Vec<DVec2> = self.coll.as_ref().unwrap().points.deref().clone()
            .iter().map(|p| { DVec2::from_angle(self.ori).rotate(*p)+my_pos }).collect();
        let other_points:Vec<DVec2> = body.coll.as_ref().unwrap().points.deref().clone()
            .iter().map(|p| { DVec2::from_angle(body.ori).rotate(*p)+other_pos }).collect();
        // first check one side, and if not found from the other side
        let mut intersection_point = Self::is_point_inside(&my_pos, &my_points, &other_points);
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
                    //*(PAUSE.write().unwrap()) = true;
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
    pub fn boundary(&self) -> [DVec2;2] {
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
    pub fn restore_last_pos(&mut self) {
        self.pos = self.last_pos;
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
/// Polygon with is centered / relative to (0,0)
pub struct Polygon {
    pub points: Box<[DVec2]>,
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
