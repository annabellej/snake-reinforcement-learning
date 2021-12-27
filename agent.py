import numpy as np
import utils


class Agent:
    def __init__(self, actions, Ne=40, C=40, gamma=0.7):
        self.actions = actions
        self.Ne = Ne  # used in exploration function
        self.C = C
        self.gamma = gamma
        self.reset()
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()

    def train(self):
        self._train = True

    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self, model_path):
        utils.save(model_path, self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self, model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        self.points = 0
        self.s = None
        self.a = None

    def exploration_function(self, q_val, n_val):
        """
        Helper function for exploration.
        """
        if n_val < self.Ne:
            return 1.0
        else:
            return q_val

    def opt_action_explore(self, state):
        """
        Helper function to choose an optimal action using exploration policy.
        """
        exp_vals = np.zeros(4)

        for i in range(4):
            exp_vals[i] = self.exploration_function(self.Q[state][i], self.N[state][i])

        max_val = max(exp_vals)
        indices = [index for index, value in enumerate(exp_vals) if value == max_val]
        return max(indices)

    def opt_action(self, state):
        """
        Helper function to choose an optimal action based on Q-table.
        """
        q_vals = np.zeros(4)

        for i in range(4):
            q_vals[i] = self.Q[state][i]

        max_val = max(q_vals)
        indices = [index for index, value in enumerate(q_vals) if value == max_val]
        return max(indices)

    def act(self, environment, points, dead):
        """
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: chosen action between utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT
        """
        s_prime = self.generate_state(environment)

        # edge case: t=0 aka s and a are None -> no need to update Q and N tables
        update_tables = True
        if self.s is None and self.a is None:
            update_tables = False

        if self._train:  # TRAIN (explore based on exploration function + exploit based on Q table)
            if update_tables:
                # get estimate of optimal future value
                est_opt_future_val = self.opt_action(s_prime)

                # get learning rate
                alpha = self.C / (self.C + self.N[self.s][self.a])

                # get reward
                reward = -0.1
                if points - self.points == 1:
                    reward = 1
                elif dead:
                    reward = -1

                # update n-values
                self.N[self.s][self.a] += 1

                # update q-values
                self.Q[self.s][self.a] += alpha * (reward + self.gamma * self.Q[s_prime][est_opt_future_val] - self.Q[self.s][self.a])

            # choose optimal action:
            result_action = self.opt_action_explore(s_prime)

        else:  # TEST (exploit based on Q table)
            result_action = self.opt_action(s_prime)

        self.s = s_prime
        self.a = result_action
        self.points = points

        if dead:
            self.reset()

        return result_action

    def generate_state(self, environment):
        snake_head = (environment[0], environment[1])
        snake_body_coords = environment[2]
        food_coord = (environment[3], environment[4])

        food_dir_x = 0
        if food_coord[0] < snake_head[0]:
            food_dir_x = 1
        elif food_coord[0] > snake_head[0]:
            food_dir_x = 2

        food_dir_y = 0
        if food_coord[1] < snake_head[1]:
            food_dir_y = 1
        elif food_coord[1] > snake_head[1]:
            food_dir_y = 2

        adjoining_wall_x = 0
        if snake_head[0] == utils.GRID_SIZE:
            adjoining_wall_x = 1
        elif snake_head[0] == utils.DISPLAY_SIZE - 2 * utils.GRID_SIZE:
            adjoining_wall_x = 2

        adjoining_wall_y = 0
        if snake_head[1] == utils.GRID_SIZE:
            adjoining_wall_x = 1
        elif snake_head[1] == utils.DISPLAY_SIZE - 2 * utils.GRID_SIZE:
            adjoining_wall_x = 2

        adjoining_body_top = 0
        adjoining_body_bottom = 0
        adjoining_body_left = 0
        adjoining_body_right = 0

        for snake_body in snake_body_coords:
            if snake_body[0] == snake_head[0] - utils.GRID_SIZE and snake_body[1] == snake_head[1]:
                adjoining_body_left = 1
            elif snake_body[0] == snake_head[0] + utils.GRID_SIZE and snake_body[1] == snake_head[1]:
                adjoining_body_right = 1
            elif snake_body[0] == snake_head[0] and snake_body[1] == snake_head[1] - utils.GRID_SIZE:
                adjoining_body_top = 1
            elif snake_body[0] == snake_head[0] and snake_body[1] == snake_head[1] + utils.GRID_SIZE:
                adjoining_body_bottom = 1

        return (food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, adjoining_body_top, adjoining_body_bottom,
                adjoining_body_left, adjoining_body_right)
