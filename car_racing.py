import gym
import numpy as np
from numpy.linalg import norm
from gym.envs.box2d.car_racing import *
from gym.envs.box2d.car_dynamics import ENGINE_POWER
# from gym.envs.classic_control.rendering import Geom, _add_attrs
# from pyglet.gl import *


class ExtendedCarRacing(CarRacing):
    def __init__(self, init_seed, stochastic, max_pos_costs):
        super(ExtendedCarRacing, self).__init__()
        self.deterministic = not stochastic
        self.init_seed = init_seed
        self.seed(init_seed)
        self.max_pos_costs = max_pos_costs
        self.min_cost = -1000. # defined by CarRacing env. In fact, this is only the minimum if you can instantaneously do the whole track
        self.env_type = 'car'
        self.alpha_dict = {}
        self.rad_dict = {}
        self.reset()
        # Get rid of black screen! I believe this is a bug in CarRacing-v0
        self.step((-1,1,0))
        self.render()
        self.reset()

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

    def _create_track(self):
        CHECKPOINTS = 12

        # Create checkpoints
        checkpoints = []
        
        for c in range(CHECKPOINTS):
            if self.deterministic:
                if c not in self.alpha_dict: 
                    self.alpha_dict[c] = self.np_random.uniform(0, 2*math.pi*1/CHECKPOINTS)
                alph = self.alpha_dict[c]
            else:
                alph = self.np_random.uniform(0, 2*math.pi*1/CHECKPOINTS)
            alpha = 2*math.pi*c/CHECKPOINTS + alph

            if self.deterministic:
                if c not in self.rad_dict: 
                    self.rad_dict[c] = self.np_random.uniform(TRACK_RAD/3, TRACK_RAD)
                r = self.rad_dict[c]
            else:
                r = self.np_random.uniform(TRACK_RAD/3, TRACK_RAD)
            rad = r
            if c==0:
                alpha = 0
                rad = 1.5*TRACK_RAD
            if c==CHECKPOINTS-1:
                alpha = 2*math.pi*c/CHECKPOINTS
                self.start_alpha = 2*math.pi*(-0.5)/CHECKPOINTS
                rad = 1.5*TRACK_RAD
            checkpoints.append( (alpha, rad*math.cos(alpha), rad*math.sin(alpha)) )

        #print "\n".join(str(h) for h in checkpoints)
        #self.road_poly = [ (    # uncomment this to see checkpoints
        #    [ (tx,ty) for a,tx,ty in checkpoints ],
        #    (0.7,0.7,0.9) ) ]
        self.road = []

        # Go from one checkpoint to another to create track
        x, y, beta = 1.5*TRACK_RAD, 0, 0
        dest_i = 0
        laps = 0
        track = []
        no_freeze = 2500
        visited_other_side = False
        while 1:
            alpha = math.atan2(y, x)
            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2*math.pi
            while True: # Find destination from checkpoints
                failed = True
                while True:
                    dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    if dest_i % len(checkpoints) == 0: break
                if not failed: break
                alpha -= 2*math.pi
                continue
            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            dest_dx = dest_x - x  # vector towards destination
            dest_dy = dest_y - y
            proj = r1x*dest_dx + r1y*dest_dy  # destination vector projected on rad
            while beta - alpha >  1.5*math.pi: beta -= 2*math.pi
            while beta - alpha < -1.5*math.pi: beta += 2*math.pi
            prev_beta = beta
            proj *= SCALE
            if proj >  0.3: beta -= min(TRACK_TURN_RATE, abs(0.001*proj))
            if proj < -0.3: beta += min(TRACK_TURN_RATE, abs(0.001*proj))
            x += p1x*TRACK_DETAIL_STEP
            y += p1y*TRACK_DETAIL_STEP
            track.append( (alpha,prev_beta*0.5 + beta*0.5,x,y) )
            if laps > 4: break
            no_freeze -= 1
            if no_freeze==0: break
        #print "\n".join([str(t) for t in enumerate(track)])

        # Find closed loop range i1..i2, first loop should be ignored, second is OK
        i1, i2 = -1, -1
        i = len(track)
        while True:
            i -= 1
            if i==0: return False  # Failed
            pass_through_start = track[i][0] > self.start_alpha and track[i-1][0] <= self.start_alpha
            if pass_through_start and i2==-1:
                i2 = i
            elif pass_through_start and i1==-1:
                i1 = i
                break
        print("Track generation: %i..%i -> %i-tiles track" % (i1, i2, i2-i1))
        assert i1!=-1
        assert i2!=-1

        track = track[i1:i2-1]

        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)
        # Length of perpendicular jump to put together head and tail
        well_glued_together = np.sqrt(
            np.square( first_perp_x*(track[0][2] - track[-1][2]) ) +
            np.square( first_perp_y*(track[0][3] - track[-1][3]) ))
        if well_glued_together > TRACK_DETAIL_STEP:
            return False

        # Red-white border on hard turns
        border = [False]*len(track)
        for i in range(len(track)):
            good = True
            oneside = 0
            for neg in range(BORDER_MIN_COUNT):
                beta1 = track[i-neg-0][1]
                beta2 = track[i-neg-1][1]
                good &= abs(beta1 - beta2) > TRACK_TURN_RATE*0.2
                oneside += np.sign(beta1 - beta2)
            good &= abs(oneside) == BORDER_MIN_COUNT
            border[i] = good
        for i in range(len(track)):
            for neg in range(BORDER_MIN_COUNT):
                border[i-neg] |= border[i]

        # Create tiles
        for i in range(len(track)):
            alpha1, beta1, x1, y1 = track[i]
            alpha2, beta2, x2, y2 = track[i-1]
            road1_l = (x1 - TRACK_WIDTH*math.cos(beta1), y1 - TRACK_WIDTH*math.sin(beta1))
            road1_r = (x1 + TRACK_WIDTH*math.cos(beta1), y1 + TRACK_WIDTH*math.sin(beta1))
            road2_l = (x2 - TRACK_WIDTH*math.cos(beta2), y2 - TRACK_WIDTH*math.sin(beta2))
            road2_r = (x2 + TRACK_WIDTH*math.cos(beta2), y2 + TRACK_WIDTH*math.sin(beta2))
            t = self.world.CreateStaticBody( fixtures = fixtureDef(
                shape=polygonShape(vertices=[road1_l, road1_r, road2_r, road2_l])
                ))
            t.userData = t
            c = 0.01*(i%3)
            t.color = [ROAD_COLOR[0] + c, ROAD_COLOR[1] + c, ROAD_COLOR[2] + c]
            t.road_visited = False
            t.road_friction = 1.0
            t.fixtures[0].sensor = True
            self.road_poly.append(( [road1_l, road1_r, road2_r, road2_l], t.color ))
            self.road.append(t)
            if border[i]:
                side = np.sign(beta2 - beta1)
                b1_l = (x1 + side* TRACK_WIDTH        *math.cos(beta1), y1 + side* TRACK_WIDTH        *math.sin(beta1))
                b1_r = (x1 + side*(TRACK_WIDTH+BORDER)*math.cos(beta1), y1 + side*(TRACK_WIDTH+BORDER)*math.sin(beta1))
                b2_l = (x2 + side* TRACK_WIDTH        *math.cos(beta2), y2 + side* TRACK_WIDTH        *math.sin(beta2))
                b2_r = (x2 + side*(TRACK_WIDTH+BORDER)*math.cos(beta2), y2 + side*(TRACK_WIDTH+BORDER)*math.sin(beta2))
                self.road_poly.append(( [b1_l, b1_r, b2_r, b2_l], (1,1,1) if i%2==0 else (1,0,0) ))
        self.track = track
        return True

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
        self.number_of_times_brake = 0
        self.deviations_from_center = []

        # if self.deterministic:
        #     st0 = np.random.get_state()
        #     self.seed(self.init_seed)

        while True:
            success = self._create_track()
            if success: break
            print("retry to generate track (normal if there are not many of this messages)")
        self.car = Car(self.world, *self.track[0][1:4])

        # if self.deterministic:
        #     # set seed back after recreating same track
        #     np.random.set_state(st0) 

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
            self.number_of_times_brake += action[2]>0
            self.deviations_from_center.append(distance_to_track)

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

    def render(self, mode='human', render_human=False):
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
            gl.glViewport(0, 0, int(WINDOW_W), int(WINDOW_H))
            t.enable()
            self.render_road()

            ## Add distance to center of road visualization
            # for idx in range(len(self.track)):
            #     alpha1, beta1, x2, y2 = self.track[idx]
            #     alpha2, beta2, x1, y1 = self.track[idx-1]

            #     # Center line of road
            #     self.viewer.draw_line((x1,y1),(x2,y2), color=(0,1,0))
            ##Line from car to center-line of road
            # if self.closest_track_point_to_hull is not None:
            #     self.viewer.draw_line(self.closest_track_point_to_hull ,(self.car.hull.position.x,self.car.hull.position.y), color=(0,0,1), width=5)
            #     self.draw_point(self.viewer, self.closest_track_point_to_hull, color=(0,0,1))

            for geom in self.viewer.onetime_geoms:
                geom.render()
            t.disable()
            self.render_indicators(WINDOW_W, WINDOW_H)
            if render_human:
                self.viewer.draw_polygon([(415,WINDOW_H-35),(860,WINDOW_H-35), (860,WINDOW_H-115), (415,WINDOW_H-115)], filled=False, color=(1,1,1), width =10)
                self.viewer.onetime_geoms[-1].render()

                tile_label = pyglet.text.Label('Number of Tiles Collected: ', font_size=18,
                x=425, y=WINDOW_H-50, anchor_x='left', anchor_y='center',
                color=(255,255,255,255))
                tile_label.draw()

                tile_label = pyglet.text.Label('0000', font_size=18,
                x=750, y=WINDOW_H-50, anchor_x='left', anchor_y='center',
                color=(255,255,255,255))
                tile_label.text = "%04i" % self.tile_visited_count
                tile_label.draw()

                brake_label = pyglet.text.Label('Number of Braking Actions: ', font_size=18,
                x=425, y=WINDOW_H-75, anchor_x='left', anchor_y='center',
                color=(255,255,255,255))
                brake_label.draw()

                brake_label = pyglet.text.Label('0000', font_size=18,
                x=750, y=WINDOW_H-75, anchor_x='left', anchor_y='center',
                color=(255,255,255,255))
                brake_label.text = "%04i" % self.number_of_times_brake
                brake_label.draw()

                lane_label = pyglet.text.Label('Mean Deviation from Center: ', font_size=18,
                x=425, y=WINDOW_H-100, anchor_x='left', anchor_y='center',
                color=(255,255,255,255))
                lane_label.draw()

                lane_label = pyglet.text.Label('0000', font_size=18,
                x=750, y=WINDOW_H-100, anchor_x='left', anchor_y='center',
                color=(255,255,255,255))


                if len(self.deviations_from_center) > 1:
                    lane_label.text = "%04f" % np.mean(self.deviations_from_center)
                else:
                    lane_label.text = "%04f" % 0
                lane_label.draw()

                #self.render_indicators(WINDOW_W, WINDOW_H)  # TODO: find why 2x needed, wtf
                image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
                arr2 = np.fromstring(image_data.data, dtype=np.uint8, sep='')
                #import pdb; pdb.set_trace()
                arr2 = arr2.reshape(WINDOW_H, WINDOW_W, 4)
                arr2 = arr2[::-1, :, 0:3]

                #win.flip()
                self.viewer.onetime_geoms = []
                return arr2
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

