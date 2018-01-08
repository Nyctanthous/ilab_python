from math import sqrt
import math

import os
import glob
import pandas as pd
from uncertainties import ufloat, unumpy
from uncertainties.umath import *


# ---------------------- Assorted helper functions -------------------------
# See http://en.wikipedia.org/wiki/Ramer-Douglas-Peucker_algorithm for
# the MATLAB version of this code.

def point_line_distance(point, start, end):
    if start == end:
        return sqrt((point[0] - start[0])**2 + (point[1] - start[1])**2)
    else:
        delta_x, delta_y = end[0] - start[0], end[1] - start[1]

        n = abs(delta_x * (start[1] - point[1]) - (start[0] - point[0]) * delta_y)
        return n / sqrt(delta_x**2 + delta_y**2)

def rdp(points, epsilon):
    dmax = 0.0
    index = 0
    for i in range(1, len(points) - 1):
        d = point_line_distance(points[i], points[0], points[-1])
        if d > dmax:
            index = i
            dmax = d

    return rdp(points[:index+1], epsilon)[:-1] + rdp(points[index:], epsilon)\
        if dmax >= epsilon else [points[0], points[-1]]

def calc_velocity(track_df, px_per_mm, fps):

    # Convert px per mm to meter(s).
    x_disp = [ufloat(element, 2)/(px_per_mm*1000) for element in track_df.x]
    y_disp = [ufloat(element, 2)/(px_per_mm*1000) for element in track_df.y]

    time_arr = [(track_df.iloc[i, 0] - track_df.iloc[i-1, 0])/fps for i in range(1, len(track_df))]

    pos_y = [y_disp[i] - y_disp[i - 1] for i in range(1, len(track_df))]
    velocity_arr = [y_pos/time for y_pos, time in zip(pos_y, time_arr)]

    # Cut off the first and last half-second to reduce
    # concerns about turning point velocities.
    half_second = int(fps/2)
    velocity_arr = velocity_arr[half_second:-half_second]
    return sum(velocity_arr)/len(velocity_arr)

def calc_charge(v_f, v_r):
    b = 0.0082    # Pascal Meters
    g = 9.80665  # meters per second squared
    eta = ufloat(0.0000184, 0.0000004)  # viscosity of air in Ns/m^2.
    # This comes from Appendix A, using a temp. of 23C.

    # Quantity in hectopascals, converting to pascals.
    p = ufloat(1025.26, 0.1) * 100

    # Measured, units in volts
    plate_diff = ufloat(498, 2)

    # Caliper is in inches, so convert to meters.
    plate_spacing = (ufloat(0.30, 0.001)*2.54)/100
    rho = ufloat(863.4, 10.2)

    E = plate_diff/plate_spacing

    # Equations
    b_over_2p = b/(2*p)
    sec_term = (9*eta*v_f)/(2*g*rho)

    a = sqrt(b_over_2p*b_over_2p + sec_term) - b_over_2p
    m = (4*math.pi*(a**3)*rho)/3
    q = -(m*g*(abs(v_f - v_r)))/(E*v_f)

    # Returns a in meters, mass in (grams?), and charge in coulombs
    return a, m, q

def analyze_df(df_fall, df_rise, px_per_mm, fps, tolerance=0.2):
    if len(df_fall) and len(df_rise):
        v_fall = calc_velocity(df_fall, px_per_mm, fps)
        v_rise = calc_velocity(df_rise, px_per_mm, fps)

        # Handle math domain issues
        if v_fall < 0 and v_rise > 0:
            # That's fine, the inputs are just flipped.
            temp = v_rise
            v_rise = v_fall
            v_fall = temp

        # Now, check if the velocities have the same inputs, and fall > rise.
        # This can be used to provide error;
        # there is a charge s.t. a particle still falls,
        # but it doesn't fall as fast. As our aparratus has
        # been constructed and demonstrated s.t. a 1 electron charge will rise,
        # this still-falling particle means something odd is acting on it,
        # be electrons gained during fall, or another source.
        max_tolerance = v_fall + tolerance*v_fall
        min_tolerance = v_fall - tolerance*v_fall


        # Check if that ratio is within the percent tolerance.
        if max_tolerance > v_rise > min_tolerance:
            return False, calc_charge(v_fall, v_rise)[-1]
        elif v_fall < 0 and v_rise < 0:
            # Our experiment was set up s.t. we are dealing with negatively charged particles.
            # Flag the positively charged particles.
            return False, calc_charge(abs(v_fall), v_rise)[-1]
        return True, calc_charge(v_fall, v_rise)[-1]

def load_tracks(addr: str):
    data_arr = []
    for file in glob.glob(addr):
        print("Opening ", file)
        temp_df = pd.read_csv(file, sep='\t', index_col='frame')

        for element in temp_df['particle'].unique():
            data_arr.append(temp_df.loc[temp_df['particle'] == element])
    return data_arr

def load_all_trajectories(base_addr: str):
    # Takes time, in seconds, and converts it to a frame index.
    t = lambda x: int(x * 30)

    get_path = lambda given_root, given_folder: os.path.join(base_addr,
                                                             given_root,
                                                             given_folder,
                                                             'track*.csv')

    all_runs = [load_tracks(get_path('Trajectories', 'df14')),
                load_tracks(get_path('Trajectories', 'df13')),
                load_tracks(get_path('Trajectories', 'df12')),
                load_tracks(get_path('Trajectories', 'df11')),
                load_tracks(get_path('Trajectories', 'df9')),
                load_tracks(get_path('Trajectories', 'df8')),
                load_tracks(get_path('Trajectories', 'df7')),
                load_tracks(get_path('Trajectories', 'df2')),
                load_tracks(get_path('Trajectories', 'df3')),
                load_tracks(get_path('Trajectories', 'df4')),
                load_tracks(get_path('Trajectories', 'df5')),
                load_tracks(get_path('Trajectories', 'df6')),
                load_tracks(get_path('vids', 'df15')),
                load_tracks(get_path('vids', 'df16')),
                load_tracks(get_path('vids', 'df17')),
                load_tracks(get_path('vids', 'df18')),
                load_tracks(get_path('vids', 'df19'))]

    # A list of all the turning points, in seconds.
    timing_list = [(3, 10, 20, 25, 41, 45, 55, 58),  #14
                   (5, 10, 20, 25, 33, 40, 45, 50, 57, 62, 67), #13
                   (5, 12, 25, 34, 41, 47),  #12
                   (3, 14, 20, 30, 35, 44, 50, 55),  #11
                   (5, 13, 22, 29, 43, 49), #9
                   (7, 11, 20, 25, 35, 39, 45, 50), #8
                   (4, 9, 28, 30, 40, 43, 55, 57), #7
                   (6, 17, 29, 34, 44, 48, 55, 60.5),  #2
                   (6, 10, 26, 29),  #3
                   (4, 17, 31, 41, 52, 60, 71, 79), #4
                   (5, 11, 17, 20, 35, 41, 50, 56, 65, 70),  #5
                   (5, 11, 30.5, 33, 47, 50, 65, 68.5),  #6
                   (4, 9, 19, 27),  #15
                   (6, 21, 34, 41),  #16
                   (6.2, 12, 27.2, 34, 45.5, 48),  #17
                   (7.8, 11.5, 24, 30, 38, 45.5),  #18
                   (2, 6.8, 25.6, 34.2, 43, 46)]  #19

    arr_to_ret = []
    builder = TupleBuilder()
    for arr_of_df, time_arr in zip(all_runs, timing_list):
        for temp_df in arr_of_df:
            # For every DF, segment it into all possible divisions.
            # Add it to our dict if the slice is not empty.
            
            if not len(time_arr):
                print("Bad time array provided!")
                continue

            # Would pull out elements, not indexes, but we have to do
            # pseudo-fancy indexing, so not happening.
            for i, curr in enumerate(time_arr):

                if i == 0:
                    sliced_df = temp_df.loc[:t(curr)]
                else:
                    sliced_df = temp_df.loc[t(time_arr[i-1]):t(curr)]

                if len(sliced_df):
                    result = builder.build_tuple(sliced_df)
                    if result is not None:
                       arr_to_ret.append(result)   

            # Now, handle the end case
            sliced_df = temp_df.loc[t(time_arr[-1]):]
            if len(sliced_df):
                result = builder.build_tuple(sliced_df)
                if result is not None:
                    arr_to_ret.append(result)

            # Reset for next time
            builder.reset()

    return arr_to_ret

class TupleBuilder:
    def __init__(self):
        self.last_tuple_state = []

    def reset(self):
        self.last_tuple_state = []

    def build_tuple(self, input):
        self.last_tuple_state.append(input)
        if len(self.last_tuple_state) == 2:
            # Full, return and reset to only have this element.
            handle = self.last_tuple_state
            self.last_tuple_state = [input]
            return handle
        else:
            # Nothing to return yet, so pass back None
            # explicitly to indicate we're not done yet.
            return None
