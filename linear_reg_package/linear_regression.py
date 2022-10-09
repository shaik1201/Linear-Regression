import numpy as np
from numpy.linalg import inv
np.set_printoptions(suppress=True)


class linear_regression:
    # given samples in array, for example: arr = [((5,10),13), ((2,6),7), ((6,4),15), ((8,8),19)]
    # return the linear regression coefficients
    
    def __init__(self, arr):
        # shape of A
        # n rows, m columns
        self.n = len(arr)
        self.m = len(arr[0][0]) + 1 # +1 for the 1 in the end of each row
        self.A = np.zeros(shape=(self.n,self.m))
        
        self.y_cords = np.array([i[1] for i in arr])
        
        self.x_cords = [list(i[0]) for i in arr]
        
        for i in self.x_cords:
            i.append(1)
        
        self.A = np.array(self.x_cords)
        self.A_T = np.transpose(self.A)
        self.inv_At_A = (inv(np.matmul(self.A_T, self.A)))
        self.At_y = self.A_T.dot(self.y_cords)
        self.coef = self.inv_At_A.dot(self.At_y)