"""
__author__  = Willy Fitra Hendria
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

discount_param = 0.9
theta=0.00001
ROW_MAX = 9
COL_MAX = 9

states = {
			(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),
			(1,0),(1,1),(1,2),(1,3),(1,4),(1,6),(1,7),(1,8),
			(2,0),(2,7),(2,8),
			(3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(3,6),(3,8),
			(4,0),(4,1),(4,7),(4,8),
			(5,0),(5,1),(5,2),(5,3),(5,4),(5,5),(5,6),(5,8),
			(6,0),(6,8),
			(7,0),(7,1),(7,2),(7,3),(7,5),(7,6),(7,7),
			(8,0),(8,1),(8,2),(8,3),(8,4),(8,5),(8,6),(8,7)
		}

rewards = {
			(0,0):5, (0,1):5, (0,2):5, (0,3):-1, (0,4):5, (0,5):5, (0,6):-1, (0,7):5, (0,8):5,
			(1,0):-1, (1,1):-1, (1,2):-1, (1,3):-1, (1,4):-1, (1,5):-20, (1,6):-1, (1,7):-1, (1,8):5,
			(2,0):-1, (2,1):-20, (2,2):-20, (2,3):-20, (2,4):-20, (2,5):-20, (2,6):-20, (2,7):-1, (2,8):-1,
			(3,0):-1, (3,1):-1, (3,2):-1, (3,3):5, (3,4):5, (3,5):5, (3,6):-1, (3,7):-20, (3,8):-1,
			(4,0):-1, (4,1):-1, (4,2):-20, (4,3):-20, (4,4):-20, (4,5):-20, (4,6):-20, (4,7):-1, (4,8):-1,
			(5,0):-1, (5,1):-1, (5,2):-1, (5,3):-1, (5,4):-1, (5,5):-1, (5,6):-1, (5,7):-20, (5,8):-1,
			(6,0):-1, (6,1):-20, (6,2):-20, (6,3):-20, (6,4):-20, (6,5):-20, (6,6):-20, (6,7):100, (6,8):-1,
			(7,0):-1, (7,1):5, (7,2):5, (7,3):5, (7,4):-20, (7,5):5, (7,6):5, (7,7):-1, (7,8):-20,
			(8,0):-1, (8,1):5, (8,2):5, (8,3):5, (8,4):-1, (8,5):5, (8,6):5, (8,7):-1, (8,8):-20
		}

def policy_evaluation(policy, is_deterministic=True):
	""" evaluate policy, and return the value function
	"""
	V = {}
	for i in range(ROW_MAX):
		for j in range(COL_MAX):
			V[(i,j)] = 0
	while True:
		delta = 0
		for s in states:
			v = 0
			for a, action_prob in enumerate(policy[s]):
				reward_and_next_state = get_reward_and_next_state(s, a, is_deterministic)
				for prob, reward, next_state in reward_and_next_state:
					v += action_prob * prob * (reward + discount_param * V[next_state])
			delta = max(delta, np.abs(v - V[s]))
			V[s] = v
		if delta < theta:
			break
	return V

def policy_iteration(policy, is_deterministic=True):
	""" improving policy by greedy, and return the optimal policy and value function
	"""
	while True:
		V = policy_evaluation(policy, is_deterministic)
		policy_stable = True
		for s in states:
			b = np.argmax(policy[s])
			n_actions = len(policy[s])
			action_values = [0 for i in range(n_actions)] # expected value of each action
			for a in range(n_actions):
				reward_and_next_state = get_reward_and_next_state(s, a, is_deterministic)
				for prob, reward, next_state in reward_and_next_state:
					action_values[a] += prob*(reward + discount_param * V[next_state])
			optimal_a = np.argmax(action_values)
			if (b != optimal_a):
				policy_stable = False
			for a in range(n_actions):
				policy[s][a] = 0 if (a != optimal_a) else 1
			
		if (policy_stable):
			return policy, V
					

	
def get_reward_and_next_state(s, a, is_deterministic):
	""" get all reward and next states for a given action a and state s
	is_deterministic is used to handle non-deterministic case
	"""
	if (a == 0): # left
		row = s[0]
		col = s[1] - 1
		if (not is_deterministic): # 45 deg to the left or to the right
			row_dev_left = s[0] + 1
			col_dev_left = s[1] - 1
			row_dev_right = s[0] - 1
			col_dev_right = s[1] - 1
	elif (a == 1): # up
		row = s[0] - 1
		col = s[1]
		if (not is_deterministic):
			row_dev_left = s[0] - 1
			col_dev_left = s[1] - 1
			row_dev_right = s[0] - 1
			col_dev_right = s[1] + 1
	elif (a == 2): # right
		row = s[0]
		col = s[1] + 1
		if (not is_deterministic):
			row_dev_left = s[0] - 1
			col_dev_left = s[1] + 1
			row_dev_right = s[0] + 1
			col_dev_right = s[1] + 1
	elif (a == 3): # down
		row = s[0] + 1
		col = s[1]
		if (not is_deterministic):
			row_dev_left = s[0] + 1
			col_dev_left = s[1] + 1
			row_dev_right = s[0] + 1
			col_dev_right = s[1] - 1
	elif (a == 4): # up left
		row = s[0] - 1
		col = s[1] - 1
		if (not is_deterministic):
			row_dev_left = s[0]
			col_dev_left = s[1] - 1
			row_dev_right = s[0] - 1
			col_dev_right = s[1]
	elif (a == 5): # up right
		row = s[0] - 1
		col = s[1] + 1
		if (not is_deterministic):
			row_dev_left = s[0] - 1
			col_dev_left = s[1]
			row_dev_right = s[0]
			col_dev_right = s[1] + 1
	elif (a == 6): # down right
		row = s[0] + 1
		col = s[1] + 1
		if (not is_deterministic):
			row_dev_left = s[0]
			col_dev_left = s[1] + 1
			row_dev_right = s[0] + 1
			col_dev_right = s[1]
	elif (a == 7): # down left
		row = s[0] + 1
		col = s[1] - 1
		if (not is_deterministic):
			row_dev_left = s[0] + 1
			col_dev_left = s[1]
			row_dev_right = s[0]
			col_dev_right = s[1] - 1
	
	# attempts to leave the grid or enter X
	if (not (row,col) in rewards or rewards[(row, col)] == -20):
		if (not (row,col) in rewards):
			reward = -5
		else:
			reward = -20
		row = s[0]
		col = s[1]
	else:
		reward = rewards[(row, col)]
	next_state = (row, col)
	
	if (not is_deterministic):
		# attempts to leave the grid or enter X
		if (not (row_dev_left,col_dev_left) in rewards or rewards[(row_dev_left, col_dev_left)] == -20):
			if (not (row_dev_left,col_dev_left) in rewards):
				reward_dev_left = -5
			else:
				reward_dev_left = -20
			row_dev_left = s[0]
			col_dev_left = s[1]
		else:
			reward_dev_left = rewards[(row_dev_left, col_dev_left)]
		next_state_dev_left = (row_dev_left, col_dev_left)
		
		
		if (not (row_dev_right,col_dev_right) in rewards or rewards[(row_dev_right, col_dev_right)] == -20):
			if (not (row_dev_right,col_dev_right) in rewards):
				reward_dev_right = -5
			else:
				reward_dev_right = -20
			row_dev_right = s[0]
			col_dev_right = s[1]
		else:
			reward_dev_right = rewards[(row_dev_right, col_dev_right)]
		next_state_dev_right = (row_dev_right, col_dev_right)
	
	if (is_deterministic):
		reward_and_next_state = [(1,reward,next_state)]
	else:
		reward_and_next_state = [(0.75, reward, next_state), (0.15, reward_dev_left, next_state_dev_left), (0.15, reward_dev_right, next_state_dev_right)]
	return reward_and_next_state
		

		
def display_value_function(V):
	""" display value function to the console
	"""
	for i in range(ROW_MAX):
		for j in range(COL_MAX):
			print(V[i,j] , end=" ")
		print()

def display_arrow_policy(policy):
	""" display policy using matplotlib
	"""
	labels_int = [[0 for i in range(COL_MAX)] for j in range(ROW_MAX)]
	for i in range(ROW_MAX):
		for j in range(COL_MAX):
			if (i == 6 and j == 7):
				labels_int[i][j] = 9
			elif (i,j) in policy:
				a = np.argmax(policy[i,j])
				labels_int[i][j] = a
			else:
				labels_int[i][j] = 8
		print()
	cmap = colors.ListedColormap(['white'])

	fig, ax = plt.subplots()
	ax.imshow(labels_int, cmap=cmap)
	ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
	ax.set_xticks(np.arange(-.5, 10, 1));
	ax.set_xticklabels([])
	ax.set_yticks(np.arange(-.5, 10, 1));
	ax.set_yticklabels([])
	ax.axis('image')

	for (j,i),label in np.ndenumerate(labels_int):
		ax.text(i,j,get_symbol(label),ha='center',va='center')

	plt.show()
					
def get_symbol(a):
	""" get symbol from a given integer
	"""
	if (a == 0):
		symbol = '←'
	elif (a == 1):
		symbol = '↑'
	elif (a == 2):
		symbol = '→'
	elif (a == 3):
		symbol = '↓'
	elif (a == 4):
		symbol = '↖  '
	elif (a == 5):
		symbol = '↗ '
	elif (a == 6):
		symbol = '↘ '
	elif (a == 7):
		symbol = '↙'
	elif (a == 8):
		symbol = 'X'
	else:
		symbol = 'G'
	return symbol


	
print("Created by:")
print("Willy Fitra Hendria")
print("------------------------------------------")
print()
print("1. Expected value of all cells for a policy that chooses with probability 0.5 a random action and otherwise moves down")
print("2. Optimal value with 4 actions");
print("3. Optimal value with 8 actions (with diagonal actions)");
print("4. Optimal value with 8 actions, non-deterministic");
print()
i = input("Input the implementation (1 - 4):")
print()
print("------------------------------------------")
print()

if i == '1':
	policy = {}
	#  probability 0.5 a random action and otherwise moves down.
	# left, up, right, down
	for s in states:
		policy[s] = [0.125, 0.125, 0.125, 0.625]

	V = policy_evaluation(policy)
	display_value_function(V)
	
elif i=='2':
	policy = {}
	#  probability 0.5 a random action and otherwise moves down.
	# left, up, right, down
	for s in states:
		policy[s] = [0.125, 0.125, 0.125, 0.625]
	policy, V = policy_iteration(policy)
	display_value_function(V)
	display_arrow_policy(policy)

elif i=='3':
	random_policy = {}
	#  uniform probability in 8 directions policy.
	# left, up, right, down, up-left, up-right, down-right, down-left
	for s in states:
		random_policy[s] = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
	policy_deterministic_diagonal, V = policy_iteration(random_policy)
	display_value_function(V)
	display_arrow_policy(policy_deterministic_diagonal)

elif i=='4':
	random_policy = {}
	#  uniform probability in 8 directions policy.
	# left, up, right, down, up-left, up-right, down-right, down-left
	for s in states:
		random_policy[s] = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
	policy_nondeterministic_diagonal, V = policy_iteration(random_policy, False)
	display_value_function(V)
	display_arrow_policy(policy_nondeterministic_diagonal)

else:
	print("Input not valid");
