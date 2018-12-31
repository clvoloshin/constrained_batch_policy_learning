import gym
import numpy as np
from numpy.linalg import norm
from gym.envs.box2d.car_racing import *
from gym.envs.box2d.car_dynamics import ENGINE_POWER
from gym.envs.classic_control.rendering import Geom, _add_attrs
from pyglet.gl import *


class ExtendedCarRacing(CarRacing):
    def __init__(self, init_seed, stochastic, max_pos_costs):
        super(ExtendedCarRacing, self).__init__()
        self.deterministic = not stochastic
        self.init_seed = init_seed
        self.max_pos_costs = max_pos_costs
        self.min_cost = -1000. # defined by CarRacing env. In fact, this is only the minimum if you can instantaneously do the whole track
        self.env_type = 'car'

    def is_early_episode_termination(self, cost=None, time_steps=None, total_cost=None):
        if cost > 0:
            self.pos_cost_counter += 1
        else:
            self.pos_cost_counter = 0

        if (self.pos_cost_counter > self.max_pos_costs) and total_cost >= -500:
            punish = 20
        else:
            punish = 0

        return (self.pos_cost_counter > self.max_pos_costs), punish

    def reset(self):
        self._destroy()
        self.amount_of_time_spent_doing_nothing = 0
        self.reward = 0.0
        self.prev_reward = 0.0
        self.prev_distance_to_track = 0.0
        self.prev_fuel = 0.0
        self.closest_track_point_to_hull = None
        self.prev_velocity_x = 0.
        self.prev_velocity_y = 0.

        self.tile_visited_count = 0
        self.t = 0.0
        self.pos_cost_counter = 0
        self.road_poly = []
        self.human_render = False
        if self.deterministic:
            st0 = np.random.get_state()
            self.seed(self.init_seed)

        while True:
            success = self._create_track()
            if success: break
            print("retry to generate track (normal if there are not many of this messages)")
        self.car = Car(self.world, *self.track[0][1:4])

        if self.deterministic:
            # set seed back after recreating same track
            np.random.set_state(st0) 

        return self.step(None)[0]

    def step(self, action):
        if action is not None:
            self.car.steer(-action[0])
            self.car.gas(action[1])
            self.car.brake(action[2])

        dt = 1.0/FPS
        self.car.step(dt)
        self.world.Step(dt, 6*30, 2*30)
        self.t += dt

        self.state = self.render("state_pixels")

        step_reward = 0
        step_fuel = 0
        step_acc = 0
        c = 0
        g = [0.]
        
        done = False
        if action is not None: # First step without action, called from reset()
            # Distance to center of track
            distances_arr = MinList()
            
            p0 = np.array([self.car.hull.position.x,self.car.hull.position.y])
            for idx in range(len(self.track)):
                alpha1, beta1, x2, y2 = self.track[idx]
                alpha2, beta2, x1, y1 = self.track[idx-1]

                # self.viewer.draw_line((x1,y1),(x2,y2), color=(0,1,0))
                p1 = np.array([x1,y1])
                p2 = np.array([x2,y2])
                
                if norm(p2-p0) <= distances_arr.get_min()[0] + 10:
                    distance, point = self.distance_from_segment_to_point(p1,p2,p0)
                    distances_arr.append(distance, point)
            
            distance_to_track, self.closest_track_point_to_hull = distances_arr.get_min()

            # Acceleration
            acc_x = (self.prev_velocity_x - self.car.hull.linearVelocity[0])/dt
            acc_y = (self.prev_velocity_y - self.car.hull.linearVelocity[1])/dt
            step_acc = np.sqrt(np.square(acc_x) + np.square(acc_y))

            self.prev_velocity_x = self.car.hull.linearVelocity[0]
            self.prev_velocity_y = self.car.hull.linearVelocity[1]

            # other reward
            self.reward -= 0.1
            step_reward = self.reward - self.prev_reward
            step_fuel = self.car.fuel_spent/ENGINE_POWER - self.prev_fuel
            self.prev_reward = self.reward
            self.prev_fuel = self.car.fuel_spent/ENGINE_POWER
            self.prev_distance_to_track = distance_to_track

            # Changed to 95% of track
            if self.tile_visited_count>= .95*len(self.track):
                done = True
            x, y = self.car.hull.position
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                done = True
                step_reward = -100

            c = -step_reward
            g = [step_fuel, step_acc, distance_to_track, self.car.hull.linearVelocity[0], self.car.hull.linearVelocity[1], action[2]>0]

        return self.state, (c,g), done, {}

    # @staticmethod
    # def distance_from_line_to_point(p1, p2, p0):
    #     # get distance from p0 to line defined from p1 to p2
    #     x1,y1 = p1
    #     x2,y2 = p2
    #     x0,y0 = p0

    #     #General form: ax + by + c = 0
    #     a = (y2 - y1)
    #     b = -(x2 - x1)
    #     c = x2*y1 - y2*x1

    #     distance = np.abs(a*x0 + b*y0 + c) / np.sqrt(a**2 + b**2)
    #     closest_point = np.array([b*(b*x0-a*y0)-a*c, a*(-b*x0+a*y0)-b*c])/(a**2 + b**2)

    #     return distance, closest_point

    @staticmethod
    def distance_from_segment_to_point(A, B, P):
        """ segment line AB, point P, where each one is an array([x, y]) """
        if np.all(A == P) or np.all(B == P):
            return 0, P
        if np.arccos(np.dot((P - A) / norm(P - A), (B - A) / norm(B - A))) > np.pi / 2:
            return norm(P - A), A
        if np.arccos(np.dot((P - B) / norm(P - B), (A - B) / norm(A - B))) > np.pi / 2:
            return norm(P - B), B

        a = B-A
        b = P-A
        projection = np.dot(a, b) / norm(a)**2 * a + A
        return norm(np.cross(A-B, A-P))/norm(B-A), projection

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.score_label = pyglet.text.Label('0000', font_size=36,
                x=20, y=WINDOW_H*2.5/40.00, anchor_x='left', anchor_y='center',
                color=(255,255,255,255))
            self.transform = rendering.Transform()

        if "t" not in self.__dict__: return  # reset() not called yet

        zoom = 0.1*SCALE*max(1-self.t, 0) + ZOOM*SCALE*min(self.t, 1)   # Animate zoom first second
        zoom_state  = ZOOM*SCALE*STATE_W/WINDOW_W
        zoom_video  = ZOOM*SCALE*VIDEO_W/WINDOW_W
        scroll_x = self.car.hull.position[0]
        scroll_y = self.car.hull.position[1]
        angle = -self.car.hull.angle
        vel = self.car.hull.linearVelocity
        if np.linalg.norm(vel) > 0.5:
            angle = math.atan2(vel[0], vel[1])
        self.transform.set_scale(zoom, zoom)
        self.transform.set_translation(
            WINDOW_W/2 - (scroll_x*zoom*math.cos(angle) - scroll_y*zoom*math.sin(angle)),
            WINDOW_H/4 - (scroll_x*zoom*math.sin(angle) + scroll_y*zoom*math.cos(angle)) )
        self.transform.set_rotation(angle)

        self.car.draw(self.viewer, mode!="state_pixels")

        arr = None
        win = self.viewer.window
        # if "dispatch_events_called" not in self.__dict__ and mode == 'state_pixels': 
        #     win.dispatch_events() 
        #     self.dispatch_events_called = True
        if mode != 'state_pixels':
            win.switch_to()
            win.dispatch_events()
        if mode=="rgb_array" or mode=="state_pixels":
            win.clear()
            t = self.transform
            if mode=='rgb_array':
                VP_W = VIDEO_W
                VP_H = VIDEO_H
            else:
                VP_W = STATE_W
                VP_H = STATE_H
            gl.glViewport(0, 0, VP_W, VP_H)
            t.enable()
            self.render_road()
            for geom in self.viewer.onetime_geoms:
                geom.render()
            t.disable()
            self.render_indicators(WINDOW_W, WINDOW_H)  # TODO: find why 2x needed, wtf
            image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
            arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
            arr = arr.reshape(VP_H, VP_W, 4)
            arr = arr[::-1, :, 0:3]

        if mode=="rgb_array" and not self.human_render: # agent can call or not call env.render() itself when recording video.
            win.flip()

        if mode=='human':
            self.human_render = True
            win.clear()
            t = self.transform
            gl.glViewport(0, 0, int(WINDOW_W*2), int(WINDOW_H*2))
            t.enable()
            self.render_road()

            # Add distance to center of road visualization
            for idx in range(len(self.track)):
                alpha1, beta1, x2, y2 = self.track[idx]
                alpha2, beta2, x1, y1 = self.track[idx-1]

                # Center line of road
                self.viewer.draw_line((x1,y1),(x2,y2), color=(0,1,0))
            # Line from car to center-line of road
            if self.closest_track_point_to_hull is not None:
                self.viewer.draw_line(self.closest_track_point_to_hull ,(self.car.hull.position.x,self.car.hull.position.y), color=(0,0,1), width=5)
                self.draw_point(self.viewer, self.closest_track_point_to_hull, color=(0,0,1))

            for geom in self.viewer.onetime_geoms:
                geom.render()
            t.disable()
            self.render_indicators(WINDOW_W, WINDOW_H)
            win.flip()

        self.viewer.onetime_geoms = []
        return arr


    def draw_point(self, viewer, point, **attrs):
        '''
        Allows for one time addition of a point
        '''
        class Point_new(Geom):
            '''
            Define a point v = (x,y) = (x,y,0)
            '''
            def __init__(self, v):
                Geom.__init__(self)
                self.v = v

            def render1(self):
                '''
                Render the point
                '''
                glPointSize(10)
                glBegin(GL_POINTS)
                glVertex3f(self.v[0], self.v[1], 1.0)
                glEnd()

        geom = Point_new(point)
        _add_attrs(geom, attrs)
        viewer.add_onetime(geom)
        return geom

class MinList(object):
    def __init__(self):
        self.distances = []
        self.points = []
        self.num_elem = -1
        self.minimum = None
        self.min_idx = None

    def append(self, distance, point):
        self.distances.append(distance)
        self.points.append(point)
        self.num_elem += 1
        if self.minimum is not None:
            if distance < self.minimum:
                self.minimum = distance
                self.min_idx = self.num_elem
        else:
            self.minimum = distance
            self.min_idx = self.num_elem

    def get_min(self):
        if self.minimum is not None:
            return self.minimum, self.points[self.min_idx]
        else:
            return np.inf, None

