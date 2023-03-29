# STEERBOX LOGIC
# handles communicating with the Arduino

# THE PROTOCOL
# to be documented

class Steerbox:
    def __init__(self):
        self.curr_voltage = None

        # search for the serial port the arduino is connected to
        for f in os.listdir("/dev"):
            if f.startswith("ttyUSB") or f.startswith("tty.usbserial"):
                p = "/dev/"+f
                break
        else:
            raise Exception("could not find arduino!")
        # open the serial port
        # reads time out after half a second
        self.ser = serial.Serial(p, baudrate=115200, timeout=0.5)
    
    def interact(self, voltage, dv=None, reset=False, allow_powerup=False):
        # receive the wheel's current position, then send a voltage
        # returns the position and the sent voltage
        # dv controls how much the voltage is allowed to change this interaction

        if reset: # reset the arduino communications
            # tell hardware to stop a bunch
            self.ser.write(b'\x00\x00\x00\x00')
            # clear anything out of the serial receive buffer
            time.sleep(0.1) # ensure hardware times out
            self.ser.reset_input_buffer()

            # tell it to stop again now that we know it's alive
            self.ser.write(b'\x00\x00')

            # wait for the updated status
            _, _, error = struct.unpack("<hBB", self.ser.read(4))
            if error == 0x81 and not allow_powerup:
                # maybe it got unplugged accidentally?
                raise Exception("hardware unexpectedly just powered up!")
            # acknowledge the error
            self.ser.write(b'\x80\x00')
            self.ser.read(4)

            self.curr_voltage = 0

        this_pos, last_quadrature_errors, last_time_ms = \
            struct.unpack("<hBB", self.ser.read(4))
        
        if last_time_ms == 0x81:
            # maybe it got unplugged accidentally?
            raise Exception("hardware unexpectedly just powered up!")
        elif last_time_ms == 0x82:
            raise Exception("Communication ran over 50ms behind!")
        elif last_time_ms & 0x80:
            raise Exception("unknown error: "+str(last_time_ms & 0x7F))
        elif last_quadrature_errors > 1:
            # the hardware got a bad value from the rotation sensor and the
            # value may now be incorrect. one error is okay as it can happen on
            # startup and represents an inaccuracy of 1/11208th of a circle. not
            # a big deal.
            raise Exception("quadrature mis-step!")
        elif last_time_ms >= 20:
            # we did not get a response out to the hardware in time
            raise Exception("Communication ran behind, took {} ms!".format(
                last_time_ms))

        # convert encoder counts to fractions of a circle
        this_pos = float(this_pos)/(4*2802)

        # wheel is rotated too much, stop the experiment before damage
        if abs(this_pos) > 1.1:
            self.ser.write(b'\x00\x00\x00\x00')
            raise Exception("Position is too large!")

        voltage = min(1, max(-1, float(voltage)))
        if dv is None:
            self.curr_voltage = voltage
        else:
            # move the current voltage towards the desired voltage
            if self.curr_voltage < voltage:
                self.curr_voltage += min(dv, voltage-self.curr_voltage)
            else:
                self.curr_voltage -= min(dv, self.curr_voltage-voltage)

        # send requested motor voltage and direction
        if self.curr_voltage >= 0:
            cmd = int(self.curr_voltage*127)
        else:
            cmd = 128 + int(-self.curr_voltage*127)
        self.ser.write(bytes([cmd]))

        return this_pos, self.curr_voltage
    
    def close(self):
        self.ser.close()
        self.curr_voltage = None



# ENVIRONMENT LOGIC
# handles casting the steerbox as an environment: taking actions and producing
# states

# the items in the state tuple the network uses (and thus the Q function)
State = namedtuple('State', [
    "pos", # current position in fractions of a circle (0 = straight ahead)
    "vel", # velocity, i.e. the difference between the current and last position
    "voltage", # current voltage of the motor
               # (0 = stopped, 1 = full speed clockwise,
               #  -1 = full speed anti-clockwise)
])

class SteerboxEnv:
    def __init__(self, box):
        self.state = None
        self.last_action = None
        self.box = box

    def reset(self):
        box = self.box

        # stop the wheel and wait for it to have no velocity
        box.interact(0, reset=True)
        time.sleep(0.5)
        box.interact(0, reset=True)
        pos = ppos = box.interact(0)[0]

        # (approximately) rotate wheel to random initial position. fortunately
        # the inaccuracy just adds to the randomness!
        goal = ((2*random.random())-1)*0.5
        if pos > goal:
            while pos > goal+0.1:
                pos, _ = box.interact(-0.7, dv=0.05)
        else:
            while pos < goal-0.1:
                pos, _ = box.interact(0.7, dv=0.05)

        # stop the wheel again
        box.interact(0, reset=True)
        time.sleep(0.5)
        # get current position
        pos, _ = box.interact(0, reset=True)
        print("goal:", ppos, "->", goal, "~", pos)

        # and initialize the current state
        self.state = State(pos, 0, 0)
        self.last_voltage = 0

        return np.array(self.state)
    
    def step(self, action):
        # perform the action in the environment and return the new state
        
        # compute the desired voltage from the action
        voltage = -1 if action == 0 else 1
        # send that voltage off (and get the current position and actual voltage)
        pos, curr_voltage = self.box.interact(voltage, dv=0.1)
        # and create the new state
        self.state = State(pos=pos, vel=pos-self.state.pos,
                           voltage=self.last_voltage)
        # remember current voltage for next state, as the time that state's
        # position is measured is the time the voltage we just sent will be
        # applied!
        self.last_voltage = curr_voltage
        
        return np.array(self.state)
    
    def close(self):
        self.box.close()

# define NFQ stuff with respect to state: success, failure, cost
class SteerboxNFQ:
    def __init__(self, env):
        self.env = env
        
        self.pos_success = 0.05
        self.vel_success = 0.01
        self.pos_failure = 0.7
        self.vel_failure = 0.04
        self.step_cost = 0.001
    
    def reset(self):
        return self.env.reset()

    def step(self, action):
        state = self.env.step(action)
        pos, vel, voltage = state

        if ( # forbidden states
            (pos > self.pos_failure or pos < -self.pos_failure)
            or (vel > self.vel_failure or vel < -self.vel_failure)
        ):
            failed = True
            cost = 1
        elif ( # goal states
            -self.pos_success < pos < self.pos_success
            and -self.vel_success < vel < self.vel_success
        ):
            failed = False
            cost = 0
        else:
            failed = False
            cost = self.step_cost
            # increase cost of time steps when the network is trying to rotate
            # the wheel away from the center
            if (pos > 0 and voltage > 0) or (pos < 0 and voltage < 0): cost *= 2
            
        return state, cost, failed
            
    def close(self):
        self.env.close()

    def experience(self, get_best_action, max_steps):
        state = self.reset()
        total_cost = 0
        experiences = []
        for step in range(max_steps):
            action = get_best_action(state)
            next_state, cost, failed = self.step(action)
            experiences.append((state, action, cost, next_state, failed))
            print("State(pos={:.4f}, vel={:.4f}, voltage={:.4f})     ".format(
                  *next_state), end="\r")
            state = next_state
            total_cost += cost
            if failed:
                break
        print()
        
        return experiences, total_cost

    def generate_goal_pattern_set(self, size=200):
        # create hint to goal data by generating random states and actions in
        # the success zone and giving them a cost of 0
        goal_state_action_b = [np.array([
            np.random.uniform(-self.pos_success, self.pos_success),
            np.random.uniform(-self.vel_success, self.vel_success),
            np.random.uniform(-0.2, 0.2),
            np.random.randint(2),
        ]) for _ in range(size)]

        goal_target_q_values = np.zeros(size)

        return goal_state_action_b, goal_target_q_values

