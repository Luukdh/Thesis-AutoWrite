import numpy as np

def interpolate(stroke, length):
        '''
        This funtion takes a normal input stroke and returns a stroke 
        with the a contant length of "len" using linear interpolation.
        '''
        new_stroke = []
        print(stroke)
        for i in np.linspace(0, len(stroke) - 1, length, endpoint=False):
            new_x = stroke[int(i)][0] + (i - int(i)) * (stroke[int(i) + 1][0] - stroke[int(i)][0])
            new_y = stroke[int(i)][1] + (i - int(i)) * (stroke[int(i) + 1][1] - stroke[int(i)][1])
            new_p = stroke[int(i)][2] + (i - int(i)) * (stroke[int(i) + 1][2] - stroke[int(i)][2])
            new_stroke.append([new_x, new_y, new_p])
        return new_stroke

# stroke = [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]
# print(interpolate(stroke, 10))

print(715 % 100)
