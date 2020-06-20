import numpy as np
import itertools
import pickle
from tensorflow import keras


class Agent(object):
  """Tic-tac-toe player.

  The agent always thinks it is player 1, so it doesn't matter who goes first or how many pieces
  are on the board. It will make the action as the player with 1s on the board.

  """
  @property
  def name(self):
    return type(self).__name__

  def quality_function(self, board_state):
    """Quality function which returns a q table based on the board state. Optional if policy is implemented."""
    raise NotImplementedError

  def update(self, board_state, action, new_board_state):
    """Update the agent during training. Optional for heuristic policies."""
    pass
    
  def policy(self, board_state, training=False):
    """The agent's policy function.

    Takes in a BoardState and returns an action in that BoardState's action space. This can be
    overridden to make a q-independent policy.

    :param board_state: 
    :returns: 
    :rtype:

    """
    
    qs = self.quality_function(board_state).copy()

    # get allowable action with highest q value
    qs[np.logical_not(board_state.action_mask)] = -np.inf  # set actions not in current action space to -inf
    action = np.random.choice(np.flatnonzero(qs == qs.max()))  # does not get a random action if all q values equal
    
    return action

  def __call__(self, *args, **kwargs):
    return self.policy(*args, **kwargs)


class UserAgent(Agent):
  keypresses = {
    'q': 0,
    'w': 1,
    'e': 2,
    'a': 3,
    's': 4,
    'd': 5,
    'z': 6,
    'x': 7,
    'c': 8
  }
  
  def policy(self, board_state, **kwargs):
    print(board_state)
    while True:
      keypress = input('>>> ')
      if keypress in self.keypresses:
        action = self.keypresses[keypress]
        break
      
      if keypress.isdigit() and int(keypress) in board_state.action_space:
        action = int(keypress)
        break      

      print(f'Choose a valid move: {board_state.action_space}')
      print(board_state)

    return action


class RandomAgent(Agent):
  """Follows a random policy."""
  def quality_function(self, board):
    return np.random.uniform(size=9)


class TopLeftAgent(Agent):
  """Always chooses the topmost, leftmost square (in that order)."""
  def quality_function(self, board):
    return np.linspace(1, 0.1, num=9)
    

class QAgent(Agent):
  """Maintains a dictionary mapping boards to 9-element q table.

  The q-tables uses the board string as a key, only adding new states when encountered. This
  reduces the complexity somewhat.

  Does not identify rotationally equivalent boards. Initializes q values to 0.

  Updates according to Q-value iteration.

  """

  def __init__(self, qtables=None, gamma=0.95, alpha=0.9):
    if qtables is None:
      self.qtables = dict()
    else:
      self.qtables = qtables

    self.gamma = gamma
    self.alpha = 0.1

  def save(self, path):
    with open(path, 'wb') as file:
      pickle.dump(self.qtables, file)

  @classmethod
  def load(cls, path):
    with open(path, 'rb') as file:
      qtables = pickle.load(file)
    return cls(qtables=qtables)

  def get_initial_qtable(self):
    return 0 * np.ones(9, np.float32)
  
  def quality_function(self, board_state):
    if board_state not in self.qtables:
      self.qtables[board_state] = self.get_initial_qtable()
    return self.qtables[board_state]

  # 012
  # 345
  # 678
  rotate_action = [2, 5, 8, 1, 4, 7, 0, 3, 6]
  
  def update(self, board_state, action, new_board_state, verbose=False):
    """Update the agent using the transition.

    Note that this is not one move later but two moves later, since it's for the same agent.

    :param board_state: board_state at turn t
    :param action: action taken on turn t
    :param new_board_state: board_state at turn t+2, which contains the reward
    :returns: 
    :rtype: 

    """

    for k in range(4):          # bug here, because action isn't being rotated for k > 0
      if board_state not in self.qtables:
        self.qtables[board_state] = self.get_initial_qtable()
      if new_board_state not in self.qtables:
        self.qtables[new_board_state] = self.get_initial_qtable()
        
      if new_board_state.done:
        self.qtables[board_state][action] = new_board_state.reward
        if verbose:
          print(f'updating player {new_board_state.player} with reward {self.qtables[board_state][action]} '
                f'from action {action}, based on board\n{board_state}\n    |\n    V\n{new_board_state}')
      else:
        self.qtables[board_state][action] = ((1 - self.alpha) * self.quality_function(board_state)[action]
                                             + self.alpha * self.gamma * np.max(self.quality_function(new_board_state)))
        if verbose:
          print(f'updating player {new_board_state.player} with q value {self.qtables[board_state][action]} '
                f'from action {action}, based on board\n{board_state}\n    |\n    V\n{new_board_state}')

      board_state = board_state.rotate()
      new_board_state = new_board_state.rotate()
      action = self.rotate_action[action]
      

class EpsilonAgent(Agent):
  """Esilon-greedy agent which explores the space randomly with probability epsilon. Otherwise, it follows the q function."""
  def __init__(self, *args, epsilon=0.1, **kwargs):
    super().__init__(*args, **kwargs)
    self.epsilon = epsilon

  def policy(self, board_state, training=True):
    if training and np.random.uniform() < self.epsilon:
      actions = list(board_state.action_space)
      return actions[np.random.randint(len(actions))]
    else:
      return super().policy(board_state, training=training)


class EpsilonQAgent(EpsilonAgent, QAgent):
  pass


class BoardState(object):
  """A single state of the board."""

  def __init__(self, board=None, player=1):
    """Return a BoardState object.

    :param board: the board array. If None, board is empty.
    :param player: whose turn is it, 1 or 2. Default is player 1 (beginning of game).

    """
    if board is None:
      self.board = np.zeros((9,), dtype=np.float32)
    else:
      self.board = board
      
    self.player = player

    self.next_player = [None, 2, 1][self.player]
    self.action_mask = self.board == 0  # where actions are allowed
    self.action_space = set(np.where(self.action_mask)[0])
    self.reward, self.done, self.winner = self.get_reward()  # reward for self.player 
    
  def other_player(self):
    return BoardState(self.board, player=self.next_player)

  def copy(self):
    return BoardState(self.board, player=self.player)
    
  def do(self, action):
    assert action in self.action_space, f'{action} in {self.action_space}'

    # make the move
    board = self.board.copy()
    board[action] = self.player
    return BoardState(board, player=self.next_player)
    
  def get_reward(self):
    """Return reward and whether the game is done, as well as the winner.

    The game is done if p1 has three spots in a row or on the diagonal.

    012
    345
    678

    :returns: (reward, done, winner). If game is not done or a tie, winner is 0.
    :rtype: 

    """
    board = self.board.reshape((3, 3))
    empty = board == 0
    p1 = board == 1 
    p2 = board == 2
 
    def is_winner(p):
      return (np.any(np.all(p, axis=0))          # |
              or np.any(np.all(p, axis=1))       # --
              or np.all(np.diag(p))              # \
              or np.all(np.diag(np.fliplr(p))))  # /
    
    if is_winner(p1):
      if self.player == 1:
        return 1, True, 1
      else:
        return 0, True, 1
    elif is_winner(p2):
      if self.player == 1:
        return 0, True, 2
      else:
        return 1, True, 2
    elif np.any(empty):
      # still places to move, but no one has won, and it's not a tie.
      return 0, False, -1
    else:
      # tie game, player 2 gets more reward for a tie.
      if self.player == 1:
        return 0.1, True, 0
      else:
        return 0.5, True, 0

  def __str__(self):
    board = [[' ', 'X', 'O'][idx] for i, idx in enumerate(self.board.astype(int))]
    return """\
 {} | {} | {}
---+---+---
 {} | {} | {}
---+---+---
 {} | {} | {} """.format(*board)

  def __repr__(self):
    return f'BoardState(({self}), player={self.player})'

  def __hash__(self):
    state_string = 'T' if self.done else ''.join('{:d}'.format(int(i)) for i in self.board)
    return hash(state_string)

  def __eq__(self, other):
    return np.all(self.board == other.board)
        
  def rotate(self, k=1, inplace=True):
    """rotate the board clockwise 90 degrees k times.

    :returns: .
    :rtype: 

    """
    if k == 0:
      return self
    
    board = self.board.reshape((3, 3))
    if inplace:
      self.board = np.rot90(board, k, axes=(1, 0)).reshape(-1)
      return self
    else:
      return self.BoardState(np.rot90(board, k, axes=(1, 0)).reshape(-1), player=self.player)


class Game(object):
  """Game object which does training.

  Can have two different agents playing each other or the same agent playing itself. If the
  opponent is specified, it is a static opponent. Thus the agent is updated only on its turn, or
  every other turn.

  If the opponent is unspecified, the agent plays itself, and is updated on every turn.

  Note that in the current implementation, the two agents wouldn't actually interact, because the
  states they see are mutually exclusive.

  """
  
  def __init__(self, agent, opponent=None, agent_goes_first=True):
    """Init.

    :param agent: the main agent, which is updated during training.
    :param opponent: the opponent. If None, the agent plays itself and is updated every turn.
    :param agent_goes_first: if True, train an agent that always goes first, else go second.
    :returns: 
    :rtype: 

    """
    if opponent is None:
      self.agents = [agent, agent]
      self.update_agents = [True, True]
    elif agent_goes_first:
      self.agents = [agent, opponent]
      self.update_agents = [True, False]
    else:
      self.agents = [opponent, agent]
      self.update_agents = [False, True]

  def report_wins(self, games, wins):
    percents = wins / games * 100
    print('Played {:,d} games:\n  {:.02f}% tied\n  {:.02f}% X ({:s})\n  {:.02f}% O ({:s})'.format(
      games, percents[0],
      percents[1], self.agents[0].name,
      percents[2], self.agents[1].name))
  
  def play(self, games=1, verbose=False):
    wins = np.zeros(3, np.int)
    for game in range(games):
      board_state = BoardState()
      for t in itertools.count():
        pidx = board_state.player - 1
        if board_state.done:
          if verbose:
            winner = 'None (tie)' if board_state.winner == 0 else f'Player {board_state.winner} ({self.agents[board_state.winner - 1].name})'
            print(f'Finished game {game} in {t} turns. Winner: {winner}.')
            
          wins[board_state.winner] += 1
          break

        agent = self.agents[pidx]
        action = agent(board_state, training=False)
        board_state = board_state.do(action)

    self.report_wins(games, wins)
    return wins

  def train(self, games=1, verbose=False):
    wins = np.zeros(3, np.int)
    for game in range(games):
      if game > 0 and game % 10 == 0:
        self.report_wins(game, wins)

      # (board, action, new_board) for each player
      transitions = [[None, None, None], [None, None, None]]

      board_state = BoardState()
      for t in itertools.count():
        pidx = board_state.player - 1
        other_pidx = board_state.next_player - 1
        agent = self.agents[pidx]
        other_agent = self.agents[other_pidx]

        if t > 1:
          transitions[pidx][2] = board_state
          agent.update(*transitions[pidx])

        action = agent(board_state, training=True)
        new_board_state = board_state.do(action)
        
        # if t == 1, pidx = 1, other_pidx = 0.
        # so other_pidx is just finding out how her first move went.
        transitions[pidx][0] = board_state
        transitions[pidx][1] = action

        if new_board_state.done:
          # update the other player, who just lost or won, on the end of the game.
          transitions[other_pidx][2] = new_board_state
          other_agent.update(*transitions[other_pidx])

          transitions[pidx][2] = new_board_state.other_player()
          agent.update(*transitions[pidx])

          if verbose:
            winner = 'None (tie)' if new_board_state.winner == 0 else f'Player {new_board_state.winner} ({self.agents[new_board_state.winner - 1].name})'
            print(f'Finished game {game} in {t} turns. Winner: {winner}.')
            
          wins[new_board_state.winner] += 1
          break

        board_state = new_board_state

    self.report_wins(games, wins)
    return wins


if __name__ == '__main__':
  top_left_agent = TopLeftAgent()
  random_agent = RandomAgent()
  user = UserAgent()

  if True:
    qagent = QAgent()
    game = Game(qagent)
    game.train(1000)
    qagent.save('qagent.pkl')
  else:
    qagent = QAgent.load('qagent.pkl')

  Game(qagent, user).train(100, verbose=True)
  
