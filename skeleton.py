import time
from typing import Tuple
import gym
import random
import requests
import numpy as np
import argparse
import sys
from gym_connect_four import ConnectFourEnv
from copy import deepcopy

env: ConnectFourEnv = gym.make("ConnectFour-v0")

SERVER_ADDRESS = "https://vilde.cs.lth.se/edap01-4inarow/"
API_KEY = 'nyckel'
STIL_ID = ["ma2872en-s"] # TODO: fill this list with your stil-id's

def call_server(move):
   res = requests.post(SERVER_ADDRESS + "move",
                       data={
                           "stil_id": STIL_ID,
                           "move": move, # -1 signals the system to start a new game. any running game is counted as a loss
                           "api_key": API_KEY,
                       })
   # For safety some respose checking is done here
   if res.status_code != 200:
      print("Server gave a bad response, error code={}".format(res.status_code))
      exit()
   if not res.json()['status']:
      print("Server returned a bad status. Return message: ")
      print(res.json()['msg'])
      exit()
   return res

def check_stats():
   res = requests.post(SERVER_ADDRESS + "stats",
                       data={
                           "stil_id": STIL_ID,
                           "api_key": API_KEY,
                       })

   stats = res.json()
   return stats

"""
You can make your code work against this simple random agent
before playing against the server.
It returns a move 0-6 or -1 if it could not make a move.
To check your code for better performance, change this code to
use your own algorithm for selecting actions too
"""
def opponents_move(env):
   env.change_player() # change to oppoent
   avmoves = env.available_moves()
   if not avmoves:
      env.change_player() # change back to student before returning
      return -1

   # TODO: Optional? change this to select actions with your policy too
   # that way you get way more interesting games, and you can see if starting
   # is enough to guarrantee a win
   action = int(input("your move: "))

   state, reward, done, _ = env.step(action)
   if done:
      if reward == 1: # reward is always in current players view
         reward = -1
   env.change_player() # change back to student before returning
   return state, reward, done

"""
   TODO: Implement your min-max alpha-beta pruning algorithm here.
   Give it whatever input arguments you think are necessary
   (and change where it is called).
   The function should return a move from 0-6
   needed - rekursiv funktion som kollar moves baserat på min & max score
            - funktion som kan assistera i att räkna ut score
            - sätt att få algoritmen att välja optimalare rutor, t.ex. mitten första, maximera antal i rad, undvik rad 2.
   """
def student_move():
         
   time1 = time.time()      
   _, move = minMax(5, env, 1, -np.inf, np.inf)
   print(time.time() - time1)
   return move

def minMax(depth, env, player, alpha, beta) -> Tuple[int, int]:
      if depth == 0 or env.is_win_state():
         #print(depth)
         return score(env.board), 0
      if player == 1: #maximizing
         best_score = -np.inf
         best_move = 0
         moves = eval(env, player) #kalla på något som sorterar moves enligt poäng
         for move in moves:
            copy = deepcopy(env)
            _, _, _, _ = copy.step(move[0])
            copy.change_player()
            points, _ = minMax(depth - 1, copy, -1 * player, alpha, beta)
            if best_score < points:
               best_score = points
               best_move = move[0]
            alpha = max(alpha, points)
            if beta <= alpha: break
            
         return best_score, best_move
      
      else: #minimizing
         best_score = np.inf
         best_move = 0
         moves = eval(env, player) #kalla på något som sorterar moves enligt poäng
         for move in moves:
            copy = deepcopy(env)
            _, _, _, _ = copy.step(move[0])
            copy.change_player()
            points, _ = minMax(depth - 1, copy, -1 * player, alpha, beta)
            if best_score > points:
               best_score = points
               best_move = move[0]
            beta = min(beta, points)
            if alpha <= beta: break
            
         return best_score, best_move

def score(env) -> int:
   points = 0
   for s in range(2,5):
      points += check_inrow(env, s)
   score_grid = [[0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0],
                 [0,0,1,1,1,0,0],
                 [0,0,8,10,8,0,0],
                 [0,0,6,4,6,0,0],
                 [0,0,8,10,8,0,0]]
   for i in range(len(env)): #kollar alla rader
            for j in range(len(env[0])): #kollar alla positioner på raden
                points += score_grid[i][j] * env[i][j]#summerar värdena från x till x + range
   return points
   
def check_inrow(env, nbr) -> int:
        x = 1
        if nbr == 4:
           x = 10000000
        elif nbr == 3:
           x = 10
        points = 0
        # Test rows
        for i in range(len(env)): #kollar alla rader
            for j in range(len(env[0]) - (nbr - 1)): #kollar alla positioner på raden
                value = sum(env[i][j:j + nbr]) #summerar värdena från x till x + range
                if abs(value) == nbr:
                    points += nbr * env[i][j] * x
                    j += nbr

        # Test columns on transpose array
        reversed_board = [list(i) for i in zip(*env)]
        for i in range(len(env[0])):
            for j in range(len(env) - (nbr - 1)):
                value = sum(reversed_board[i][j:j + nbr])
                if abs(value) == nbr:
                    points += nbr * reversed_board[i][j] * x
                    j += nbr

        # Test diagonal
        for i in range(len(env) - (nbr - 1)):
            for j in range(len(env[0]) - (nbr - 1)):
                value = 0
                for k in range(nbr):
                    value += env[i + k][j + k]
                    if abs(value) == nbr:
                        points += nbr * env[i][j] * x

        reversed_board = np.fliplr(env)
        # Test reverse diagonal
        for i in range(len(env) - (nbr - 1)):
            for j in range(len(env[0]) - (nbr - 1)):
                value = 0
                for k in range(nbr):
                    value += reversed_board[i + k][j + k]
                    if abs(value) == nbr:
                        points += nbr * reversed_board[i][j] * x

        return points

def eval(board, player):
   moves = board.available_moves()
   points = 0
   order = []
   for move in moves:
      copy = deepcopy(board)
      check,_,_,_ = copy.step(move)
      points = score(check)
      order.append((move, points))
   if player == 1:
      order.sort(key=lambda a: a[1], reverse=True)
   else:
      order.sort(key=lambda a: a[1], reverse=False)  
   return order
   

def play_game(vs_server = False):
   """
   The reward for a game is as follows. You get a
   botaction = random.choice(list(avmoves)) reward from the
   server after each move, but it is 0 while the game is running
   loss = -1
   win = +1
   draw = +0.5
   error = -10 (you get this if you try to play in a full column)
   Currently the player always makes the first move
   """

   # default state
   state = np.zeros((6, 7), dtype=int)

   # setup new game
   if vs_server:
      # Start a new game
      res = call_server(-1) # -1 signals the system to start a new game. any running game is counted as a loss

      # This should tell you if you or the bot starts
      print(res.json()['msg'])
      botmove = res.json()['botmove']
      state = np.array(res.json()['state'])
      # reset env to state from the server (if you want to use it to keep track)
      env.reset(board=state)
   else:
      # reset game to starting state
      env.reset(board=None)
      # determine first player
      student_gets_move = random.choice([True, False])
      if student_gets_move:
         print('You start!')
         print()
      else:
         print('Bot starts!')
         print()

   # Print current gamestate
   print("Current state (1 are student discs, -1 are servers, 0 is empty): ")
   print(state)
   print()

   done = False
   while not done:
      # Select your move
      stmove = student_move() # TODO: change input here state, 5, -np.inf, -np.inf, True

      # make both student and bot/server moves
      if vs_server:
         # Send your move to server and get response
         res = call_server(stmove)
         print(res.json()['msg'])

         # Extract response values
         result = res.json()['result']
         botmove = res.json()['botmove']
         state = np.array(res.json()['state'])
         # reset env to state from the server (if you want to use it to keep track)
         env.reset(board=state)
      else:
         if student_gets_move:
            # Execute your move
            avmoves = env.available_moves()
            if stmove not in avmoves:
               print("You tied to make an illegal move! You have lost the game.")
               break
            state, result, done, _ = env.step(stmove)

         student_gets_move = True # student only skips move first turn if bot starts

         # print or render state here if you like

         # select and make a move for the opponent, returned reward from students view
         if not done:
            state, result, done = opponents_move(env)

      # Check if the game is over
      if result != 0:
         done = True
         if not vs_server:
            print("Game over. ", end="")
         if result == 1:
            print("You won!")
         elif result == 0.5:
            print("It's a draw!")
         elif result == -1:
            print("You lost!")
         elif result == -10:
            print("You made an illegal move and have lost!")
         else:
            print("Unexpected result result={}".format(result))
         if not vs_server:
            print("Final state (1 are student discs, -1 are servers, 0 is empty): ")
      else:
         print("Current state (1 are student discs, -1 are servers, 0 is empty): ")

      # Print current gamestate
      print(state)
      print()

def main():
   # Parse command line arguments
   parser = argparse.ArgumentParser()
   group = parser.add_mutually_exclusive_group()
   group.add_argument("-l", "--local", help = "Play locally", action="store_true")
   group.add_argument("-o", "--online", help = "Play online vs server", action="store_true")
   parser.add_argument("-s", "--stats", help = "Show your current online stats", action="store_true")
   args = parser.parse_args()

   # Print usage info if no arguments are given
   if len(sys.argv)==1:
      parser.print_help(sys.stderr)
      sys.exit(1)

   if args.local:
      play_game(vs_server = False)
   elif args.online:
      play_game(vs_server = True)

   if args.stats:
      stats = check_stats()
      print(stats)

   # TODO: Run program with "--online" when you are ready to play against the server
   # the results of your games there will be logged
   # you can check your stats bu running the program with "--stats"

if __name__ == "__main__":
    main()
