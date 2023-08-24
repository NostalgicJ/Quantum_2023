##############################################
#   
#   << Path finding by AStar Algorithm >>
# 
#     * coded by Gyuwon Ha (2023.06)
#     * commented and edited by Yeojung Jo (2023.08)
# 
#     * Phase related calculations are not determined in heuristic estimation
# 
# 
#     - global variables
# 
#         sx, sy, sz : sigma x, sigma y, sigma z
#         s0 : Identity
#         d0 : Speed of rotation on the z-axis (μs)
#         dt : Time applied per pulse (μs)
#         v0 : Spees of rotation by pulse (μs)
#         (complex number = j = (-1)**0.5)
# 
#  
#     - functions
#
#       [ Quantum State part ]
#
#         Rx(theta) : Rotation that turns as theta
#         Rz(phi) : Rotation that turns as phi
#         unitary(dt, choice) : Create a unity gate with dt and choice
#         state_fidelity(rho_1, rho_2) 
#             : Calculate the fidelity by receiving two density matrices (current state and target state)
#             ex. If fidelity is 0.99 or higher, it is considered to be a match.
#         phase_cal(x, y) : Calculate the phase information of the current state from the x,y plane
#
#       [ AStar Algorithm part ]
# 
#         Astar_Qsearch(init, target)
#             : Estimate the path (sequence combination) with the Astar algorithm 
#               by receiving the initial state and the target state.
#         heuristic(node, target) 
#             : Heuristic estimation through QSL (quantum speed limit) from current node to target
# 
# 
#     * Omitted, you can see the code below
#       [ main part ]
#       [ data part ]
#       [ starting part ]
#  
##############################################


from math import *
import numpy as np
from scipy import linalg
from scipy.linalg import expm
from qutip import *
import random
from datetime import datetime
import time
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power
from sklearn.feature_extraction.text import CountVectorizer
import heapq
import cmath
import os



#######################################
##        Quantum State part         ##
#######################################

# pauli matrix 
sx = np.array([[0,  1],     [1, 0]])
sy = np.array([[0, -1j],   [1j, 0]])
sz = np.array([[1, 0],     [0, -1]])
s0 = np.array([[1, 0],      [0, 1]])

# parameters
d0 = 0.15           # Arbitrary settings, Actual speed : 0.30rad/μs
dt = 2.6 
v0 = 0.02           # Arbitrary settings, Actual speed : 0.04rad/μs


# x-rotation operater
def Rx(theta):
    return np.matrix([  [cos(theta/2),    -1j*sin(theta/2)],
                        [-1j*sin(theta/2),    cos(theta/2)]])

# z-rotation operater
def Rz(phi): 
    return np.matrix([  [cos(phi/2)-1j*sin(phi/2),  0],
                        [0,  cos(phi/2)+1j*sin(phi/2)]])

# unitary operator
def unitary(dt, choice):
    
    # Select x,y-rotation direction.
    # [stay, +x, -x, +y, -y]
    choice_list = [0, 1, -1, 1, -1] 
    
    if choice < 3:
        # if choice = 0 ... only d0*sz
        Ham = (d0*sz+v0*choice_list[choice]*sx)
    else:
        Ham = (d0*sz+v0*choice_list[choice]*sy)

    # Creating a Unitary Operator for each of the four sections by Hamiltonian
    eigvals = np.linalg.eigh(Ham)[0]
    eigvecs = 1*np.linalg.eigh(Ham)[1]
    E = np.diag(eigvals)
    U_H = eigvecs.conj().T
    U_e = U_H.conj().T @ expm(-1j*E*dt) @ U_H
    
    return U_e


# Calculating the Fidelity
def state_fidelity(rho_1, rho_2): 
    
    # rho_1(current state), rho_2(target state)
    # Calculate the fidelity after checking the dimensions of the two states.
    
    if np.shape(rho_1) != np.shape(rho_2):
            print("Dimensions of two states do not match.")
            return 0
    else:
        sqrt_rho_1 = fractional_matrix_power(rho_1, 1 / 2)
        fidelity = np.trace(fractional_matrix_power(sqrt_rho_1 @ rho_2 @ sqrt_rho_1, 1 / 2)) ** 2
        return np.real(fidelity)

# Caculating the Phase
def phase_cal(x, y):
    
    # Calculate phase from quadrant point of view
    
    if y >= 0 and x >= 0:
        phase = atan(y/x)
    elif y > 0 and x < 0:
        phase = atan(-x/y) + pi/2
    elif y < 0 and x < 0:
        phase = atan(y/x) + pi
    else: 
        phase = atan(-x/y) + 3/2*pi
        
    return phase



#######################################
##       AStar Algorithm part        ##
#######################################

class Node:
    
    def __init__(self, parent=None, density_matrix=None, past_pulse=None):
        self.parent = parent
        self.density_matrix = np.matrix(density_matrix)
        self.past_pulse = past_pulse

        self.x = round(np.trace(self.density_matrix*sx).real, 9)
        self.y = round(np.trace(self.density_matrix*sy).real, 9)
        self.z = round(np.trace(self.density_matrix*sz).real, 9)
        
        self.phase = phase_cal(self.x, self.y)
        self.point = (self.x, self.y, self.z)
        
        self.g = 0      # Time moved to the current position 
        self.h = 0      # Estimated time to target (heuristic)
        self.f = 0      # f = g + h


# Using Astar Algorithm
def Astar_Qsearch(init, target):
    
    # A unitary operator is created in advance to reduce unnecessary operations.
    choiceList = []
    for i in range(5):
        choiceList.append(unitary(dt, i))

    # Initialize openList, closedList
    maximum_len = 100000
    openList = []
    closedList = []

    # Add initial node to openList
    heapq.heappush(openList, (init.f, init))

    # Run until fidelity 0.99 is satisfied
    iteration = 0
    while openList:


        # 원래 이산적인 node에 대한 A*는 같은 점인지 아닌지를 판단할 수 있는데, 지금 시스템에서는 같은점일 가능성이 거의 제로이다.
        # 비슷한 위치에 있는 점은 같은 점으로 인식해줘야 탐색 효율성을 높일 수 있다.
        if iteration < 10: 
            fid = 0.9999 # 초반에는 로테이션을 해도 많이 움직이지 못한다. 그런 점들을 모두 같은 점으로 인식해서는 안되므로 두 점의 fidelity가 0.9999이상일때만 같다고 인식.
        elif iteration < 800:
            fid = 0.99 # 후반에는 점들 사이의 fidelity가 0.99이상이면 같다고 인식.
        else :
            return [-1] # 800회의 iteration 내에 정답을 찾지 못하면 포기. 데이터를 많이 획득하기 위한 전략.
        

        # Get the node with the lowest f-score from the openList
        _, currentNode = heapq.heappop(openList)

        # Add currentNode to closedList
        closedList.append(currentNode)
        
        # 그냥 잘 돌아가는지 볼라고 넣음
        print(currentNode.g,currentNode.h,currentNode.f)
        
        
        if state_fidelity(currentNode.density_matrix, target.density_matrix) > 0.99:
            path = []
            current = currentNode
            while current is not None:
                path.append(current.past_pulse)
                current = current.parent
            path = path[:len(path) - 1]
            path = path[::-1]
            print(iteration)
            return path

        children = []
        i = 0
        for rotation in choiceList:
            NewDensityMatrix = rotation @ currentNode.density_matrix @ rotation.conj().T
            new_node = Node(currentNode, NewDensityMatrix, i)
            i += 1
            children.append(new_node)

        for child in children:
            value = True
            for closedNode in closedList:
                if state_fidelity(child.density_matrix, closedNode.density_matrix) > fid:
                    value = False
                    break
            if not value:
                continue
            child.g = currentNode.g + dt
            child.h = heuristic(child, target)
            child.f = child.g + child.h

            # 자식이 openList에 있고, g값이 더 크면 continue
            if any(child.f < openNode[0] and state_fidelity(child.density_matrix, openNode[1].density_matrix) > 0.999 and child.g > openNode[1].g
                   for openNode in openList):
                continue

            heapq.heappush(openList, (child.f, child))

        iteration += 1
        if len(openList) > maximum_len:
            while len(openList) > maximum_len:
                heapq.heappop(openList)

    return [-1]  # No path found

# Heuristic estimation
def heuristic(node, target):
    
    # Time Remaining from Current Node to Target
    # Consider both the difference in z-axis reference and the difference in phase (x, y-axis plane)
    
    # Difference on the z-axis
    delta_z = abs((acos(target.z) - acos(node.z)) / 0.0365)
    
    # Phase Difference
    # Not accurate and needs to be corrected, but there is no significant change in the value.
    phase = target.phase - node.phase + 0.1726*(theta-acos(node.z))
    # phase = target.phase - node.phase + 0.1726*theta

    # Adjust the value when negative.
    if phase < 0:
        phase += 2*pi

    # Arbitrary weight 0.1
    phi = phase*0.1

    h = delta_z + phi

    return h



#######################################
##             main part             ##
#######################################

def main(target_theta, target_phi) :
    
    # Initial state
    init_wave = np.array([[1], [0]])
    irho_init = np.kron(init_wave, init_wave.conj().T)
    
    # Target state
    # Theta must move first and then phi move.
    target_U = Rz(target_phi) @ Rx(target_theta) 
    irho_target = target_U @ irho_init @ target_U.conj().T

    # Finding path 
    start = Node(None, irho_init, None)
    end = Node(None, irho_target, None)
    
    # Calculating the computational time for a pair of theta and phi
    start_time = time.time()
    
    path = Astar_Qsearch(start, end)
    
    end_time = time.time()
    
    # AStar Algorithm computation time
    computing_time = end_time - start_time
    
    # Actual dt applied time
    total_time = dt*len(path)

    # Result output
    print(f"""
    ------------------------------------------------------------------------------------------------------
    theta = {target_theta}
    phi = {target_phi}
    path : {path}
    computing_time : {computing_time}
    total_time : {total_time}
    ------------------------------------------------------------------------------------------------------
    """)
    
    return path, computing_time



#######################################
##             data part             ##
#######################################

# Create a directory to store the results
dir = 'Astar_results_dir'
if not os.path.exists(dir):
    os.makedirs(dir)

# CSV file name setup
date = datetime.now()
printdate = date.strftime('%Y%m%d_%H%M%S')
filename = "/ByAstar_dt_" + str(dt) + "_" + printdate + '.csv'

# Create an empty DataFrame and write to CSV file
df = pd.DataFrame(columns=["Case", 'gate length', 'Theta', 'Phi', 'dt', 'combination', 
                           'total time', 'computing time'])
df.to_csv(dir + filename, index=False)



#######################################
##           starting part           ##
#######################################

case = 0

while True : 
    case += 1
    theta, phi = (pi/180)*random.uniform(0, 180), (pi/180)*random.uniform(0, 360)
    path, computing_time = main(theta, phi)
    output = [['case' + str(case), len(path), theta, phi, dt, path, len(path)*dt, computing_time]]
    
    # Create DataFrame and append to CSV file
    df = pd.DataFrame(output, columns=["Case", 'gate length', 'Theta', 'Phi', 'dt', 'combination', 
                                       'total time','computing time'])
    df.to_csv(filename, mode='a', header=False, index=False)


