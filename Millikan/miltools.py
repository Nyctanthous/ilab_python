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

    pos_x = [x_disp[i] - x_disp[i - 1] for i in range(1, len(track_df))]
    pos_y = [y_disp[i] - y_disp[i - 1] for i in range(1, len(track_df))]
    velocity_arr = [y_pos/time for y_pos, time in zip(pos_y, time_arr)]

    velocity_arr = velocity_arr[5:-5]
    return sum(velocity_arr)/len(velocity_arr)

def calc_charge(v_f, v_r):
    b = 0.0082    # Pascal Meters
    g = 9.80665  # meters per second squared
    eta = ufloat(0.0000184, 0.0000004)  # viscosity of air in Ns/m^2.
    # This comes from Appendix A, using a temp. of 23C.

    p = ufloat(1025.26, 0.1) * 100 # Quantity in hectopascals, converting to pascals.
    plate_diff = ufloat(498, 2) # Measured, units in volts
    plate_spacing = (ufloat(0.30, 0.001)*2.54)/100 # Caliper is in inches, so convert to meters.
    rho = ufloat(863.4, 10.2)

    E = plate_diff/plate_spacing

    # Equations
    b_over_2p = b/(2*p)
    sec_term = (9*eta*v_f)/(2*g*rho)

    a = sqrt(b_over_2p*b_over_2p + sec_term) - b_over_2p
    m = (4*math.pi*(a**3)*rho)/3
    q = (m*g*(v_f + v_r))/(E*v_f)

    # Returns a in meters, mass in (grams?), and charge in coulombs
    return a, m, q

def analyze_df(df_fall, df_rise, px_per_mm, fps):
    if len(df_fall) and len(df_rise):
        v_fall = calc_velocity(df_fall, px_per_mm, fps)
        v_rise = -calc_velocity(df_rise, px_per_mm, fps)
        if v_rise < 0:
            # Use this track to find our err from zero.
            return False, calc_charge(v_fall, -abs(v_rise))[-1]
        # Else....
        return True, calc_charge(v_fall, v_rise)[-1]
    # Else..
    return -1

def load_tracks(addr : str) -> list:
    data_arr = []
    for file in glob.glob(addr):
        print("Opening ", file)
        temp_df = pd.read_csv(file, sep='\t', index_col='frame')

        for element in temp_df['particle'].unique():
            data_arr.append(temp_df.loc[temp_df['particle'] == element])
    return data_arr

def load_all_trajectories(base_addr : str) -> dict:
    t = lambda x: int(x * 30)
    run_0, run_1, run_2 = load_tracks(base_addr + os.path.normpath('/Trajectories/df14/track*.csv'))
    run_3, run_4 = load_tracks(base_addr + os.path.normpath('/Trajectories/df13/track*.csv'))
    run_5 = load_tracks(base_addr + os.path.normpath('/Trajectories/df12/track*.csv'))[0]
    run_6, run_7, run_8, run_9 = load_tracks(base_addr + os.path.normpath('/Trajectories/df11/track*.csv'))
    run_10 = load_tracks(base_addr + os.path.normpath('/Trajectories/df9/track*.csv'))[0]
    run_11 = load_tracks(base_addr + os.path.normpath('/Trajectories/df8/track*.csv'))[0]
    run_12, run_13, run_14 = load_tracks(base_addr + os.path.normpath('/Trajectories/df7/track*.csv'))
    df2_0, df2_1, df2_2, df2_3, df2_4, df2_5, df2_6, df2_7, df2_8, df2_9 = load_tracks(base_addr + '/Trajectories/df2/track*.csv')
    df3_0, df3_1, df3_2, df3_3 = load_tracks(base_addr + '/Trajectories/df3/track*.csv')
    df4_0 = load_tracks(base_addr + '/Trajectories/df4/track*.csv')[0]
    df5_0, df5_1, df5_2, df5_3, df5_4, df5_5, df5_6, df5_7, df5_8 = load_tracks(base_addr + '/Trajectories/df5/track*.csv')
    df6_0, df6_1, df6_2, df6_3, df6_4, df6_5, df6_6, df6_7, df6_8, df6_9, df6_10 = load_tracks(base_addr + '/Trajectories/df6/track*.csv')
    df15_0, df15_1, df15_2 = load_tracks(base_addr + '/vids/df15/track*.csv')
    df16_0 = load_tracks(base_addr + '/vids/df16/track*.csv')[0]
    df17_0, df17_1, df17_2, df17_3, df17_4, df17_5, df17_6 = load_tracks(base_addr + '/vids/df17/track*.csv')
    df18_0, df18_1, df18_2, df18_3, df18_4, df18_5, df18_6, df18_7, df18_8, df18_9 = load_tracks(base_addr + '/vids/df18/track*.csv')
    df19_0, df19_1, df19_2, df19_3, df19_4, df19_5, df19_6, df19_7, df19_8, df19_9, df19_10 = load_tracks(base_addr + '/vids/df19/track*.csv')


    return {'run_0_up_0': run_0.loc[t(3):t(10)],
            'run_0_down_0': run_0.loc[t(10):t(20)],
            'run_0_up_1': run_0.loc[t(20):t(25)],
            'run_0_down_1': run_0.loc[t(25):t(41)],
            'run_0_up_2': run_0.loc[t(41):t(45)],
            'run_0_down_3': run_0.loc[t(45):t(55)],
            'run_0_up_3': run_0.loc[t(55):t(58)],
            'run_0_down_4': run_0.loc[t(58):],
            'run_1_up_0' : run_1.loc[t(3):t(10)],
            'run_1_down_0' : run_1.loc[t(10):t(20)],
            'run_1_up_1' : run_1.loc[t(20):t(25)],
            'run_1_down_1' : run_1.loc[t(25):t(41)],
            'run_1_up_2' : run_1.loc[t(41):t(45)],
            'run_1_down_3' : run_1.loc[t(45):t(55)],
            'run_1_up_3' : run_1.loc[t(55):t(58)],
            'run_1_down_4' : run_1.loc[t(58):],
            'run_2_up_0' : run_2.loc[t(3):t(10)],
            'run_2_down_0' : run_2.loc[t(10):t(20)],
            'run_2_up_1' : run_2.loc[t(20):t(25)],
            'run_2_down_1' : run_2.loc[t(25):t(41)],
            'run_2_up_2' : run_2.loc[t(41):t(45)],
            'run_2_down_3' : run_2.loc[t(45):t(55)],
            'run_2_up_3' : run_2.loc[t(55):t(58)],
            'run_2_down_4' : run_2.loc[t(58):],
            'run_3_gravity' : run_3.loc[:t(5)],
            'run_3_up_0' : run_3.loc[t(5):t(10)],
            'run_3_down_0' : run_3.loc[t(10):t(20)],
            'run_3_up_1' : run_3.loc[t(20):t(25)],
            'run_3_down_1' : run_3.loc[t(25):t(33)],
            'run_3_up_2' : run_3.loc[t(33):t(40)],
            'run_3_down_2' : run_3.loc[t(40):t(45)],
            'run_3_up_3' : run_3.loc[t(45):t(50)],
            'run_3_down_3' : run_3.loc[t(50):t(57)],
            'run_3_up_4' : run_3.loc[t(57):t(62)],
            'run_3_down_4' : run_3.loc[t(62):t(67)],
            'run_4_gravity' : run_4.loc[:t(5)],
            'run_4_up_0' : run_4.loc[t(5):t(10)],
            'run_4_down_0' : run_4.loc[t(10):t(20)],
            'run_4_up_1' : run_4.loc[t(20):t(25)],
            'run_4_down_1' : run_4.loc[t(25):t(33)],
            'run_4_up_2' : run_4.loc[t(33):t(40)],
            'run_4_down_2' : run_4.loc[t(40):t(45)],
            'run_4_up_3' : run_4.loc[t(45):t(50)],
            'run_4_down_4' : run_4.loc[t(50):t(57)],
            'run_4_up_4' : run_4.loc[t(57):t(62)],
            'run_4_down_3' : run_4.loc[t(62):t(67)],
            'run_5_gravity' : run_5.loc[:t(5)],
            'run_5_up_0' : run_5.loc[t(5):t(12)],
            'run_5_down_0' : run_5.loc[t(12):t(25)],
            'run_5_up_1' : run_5.loc[t(25):t(34)],
            'run_5_down_1' : run_5.loc[t(34):t(41)],
            'run_5_up_2' : run_5.loc[t(41):t(47)],
            'run_5_down_2' : run_5.loc[t(47):],
            'run_6_gravity' : run_6.loc[:t(3)],
            'run_6_up_0' : run_6.loc[t(3):t(14)],
            'run_6_down_0' : run_6.loc[t(14):t(20)],
            'run_6_up_1' : run_6.loc[t(20):t(30)],
            'run_6_down_1' : run_6.loc[t(30):t(35)],
            'run_6_up_2' : run_6.loc[t(35):t(44)],
            'run_6_down_2' : run_6.loc[t(44):t(50)],
            'run_6_up_3' : run_6.loc[t(50):t(55)],
            'run_6_down_3' : run_6.loc[t(55):],
            'run_7_gravity' : run_7.loc[:t(3)],
            'run_7_up_0' : run_7.loc[t(3):t(14)],
            'run_7_down_0' : run_7.loc[t(14):t(20)],
            'run_7_up_1' : run_7.loc[t(20):t(30)],
            'run_7_down_1' : run_7.loc[t(30):t(35)],
            'run_7_up_2' : run_7.loc[t(35):t(44)],
            'run_7_down_2' : run_7.loc[t(44):t(50)],
            'run_7_up_3' : run_7.loc[t(50):t(55)],
            'run_7_down_3' : run_7.loc[t(55):],
            'run_8_gravity' : run_8.loc[:t(3)],
            'run_8_up_0' : run_8.loc[t(3):t(14)],
            'run_8_down_0' : run_8.loc[t(14):t(20)],
            'run_8_up_1' : run_8.loc[t(20):t(30)],
            'run_8_down_1' : run_8.loc[t(30):t(35)],
            'run_8_up_2' : run_8.loc[t(35):t(44)],
            'run_8_down_2' : run_8.loc[t(44):t(50)],
            'run_8_up_3' : run_8.loc[t(50):t(55)],
            'run_8_down_3' : run_8.loc[t(55):],
            'run_9_gravity' : run_9.loc[:t(3)],
            'run_9_up_0' : run_9.loc[t(3):t(14)],
            'run_9_down_0' : run_9.loc[t(14):t(20)],
            'run_9_up_1' : run_9.loc[t(20):t(30)],
            'run_9_down_1' : run_9.loc[t(30):t(35)],
            'run_9_up_2' : run_9.loc[t(35):t(44)],
            'run_9_down_2' : run_9.loc[t(44):t(50)],
            'run_9_up_3' : run_9.loc[t(50):t(55)],
            'run_9_down_3' : run_9.loc[t(55):],
            'run_10_gravity' : run_10.loc[:t(5)],
            'run_10_up_0' : run_10.loc[t(5):t(13)],
            'run_10_down_0' : run_10.loc[t(13):t(22)],
            'run_10_up_1' : run_10.loc[t(22):t(29)],
            'run_10_down_1' : run_10.loc[t(29):t(43)],
            'run_10_up_2' : run_10.loc[t(43):t(49)],
            'run_10_down_2' : run_10.loc[t(49):],
            'run_11_gravity' : run_11.loc[:t(7)],
            'run_11_up_0' : run_11.loc[t(7):t(11)],
            'run_11_down_0' : run_11.loc[t(11):t(20)],
            'run_11_up_1' : run_11.loc[t(20):t(25)],
            'run_11_down_1' : run_11.loc[t(25):t(35)],
            'run_11_up_2' : run_11.loc[t(35):t(39)],
            'run_11_down_2' : run_11.loc[t(39):t(45)],
            'run_11_up_3' : run_11.loc[t(45):t(50)],
            'run_11_down_3' : run_11.loc[t(50):],
            'run_12_gravity' : run_12.loc[:t(4)],
            'run_12_up_0' : run_12.loc[:t(4):t(9)],
            'run_12_down_0' : run_12.loc[t(9):t(28)],
            'run_12_up_1' : run_12.loc[:t(28):t(30)],
            'run_12_down_1' : run_12.loc[t(30):t(40)],
            'run_12_up_2' : run_12.loc[:t(40):t(43)],
            'run_12_down_2' : run_12.loc[t(43):t(55)],
            'run_12_up_3' : run_12.loc[:t(55):t(57)],
            'run_12_down_3' : run_12.loc[t(57):],
            'run_13_gravity' : run_13.loc[:t(4)],
            'run_13_up_0' : run_13.loc[:t(4):t(9)],
            'run_13_down_0' : run_13.loc[t(9):t(28)],
            'run_13_up_1' : run_13.loc[:t(28):t(30)],
            'run_13_down_1' : run_13.loc[t(30):t(40)],
            'run_13_up_2' : run_13.loc[:t(40):t(43)],
            'run_13_down_2' : run_13.loc[t(43):t(55)],
            'run_13_up_3' : run_13.loc[:t(55):t(57)],
            'run_13_down_3' : run_13.loc[t(57):],
            'run_14_gravity' : run_14.loc[:t(4)],
            'run_14_up_0' : run_14.loc[:t(4):t(9)],
            'run_14_down_0' : run_14.loc[t(9):t(28)],
            'run_14_up_1' : run_14.loc[:t(28):t(30)],
            'run_14_down_1' : run_14.loc[t(30):t(40)],
            'run_14_up_2' : run_14.loc[:t(40):t(43)],
            'run_14_down_2' : run_14.loc[t(43):t(55)],
            'run_14_up_3' : run_14.loc[:t(55):t(57)],
            'run_14_down_3' : run_14.loc[t(57):],
            'df2_0_up_0' : df2_0.loc[t(55):t(60)],
            'df2_0_down_1' : df2_0.loc[t(60):],
            'df2_1_down_0' : df2_1.loc[:t(6)],
            'df2_1_up_0' : df2_1.loc[t(6):t(17)],
            'df2_1_down_1' : df2_1.loc[t(17):t(29)],
            'df2_1_up_1' : df2_1.loc[t(29):t(34)],
            'df2_1_down_2' : df2_1.loc[t(34):t(44)],
            'df2_2_up_0' : df2_2.loc[t(55):t(60.5)],
            'df2_2_down_1' : df2_2.loc[t(60.5):],
            'df2_3_down_0' : df2_3.loc[:t(6)],
            'df2_3_up_0' : df2_3.loc[t(6):t(17)],
            'df2_4_down_0' : df2_4.loc[t(34):t(44)],
            'df2_4_up_0' : df2_4.loc[t(44):t(48)],
            'df2_4_down_1' : df2_4.loc[t(48):t(55)],
            'df2_4_up_1' : df2_4.loc[t(55):t(60.5)],
            'df2_4_down_2' : df2_4.loc[t(60.5):],
            'df2_5_down_0' : df2_5.loc[t(16):t(29)],
            'df2_5_up_0' : df2_5.loc[t(29):t(34)],
            'df2_5_down_1' : df2_5.loc[t(34):t(44)],
            'df2_5_up_1' : df2_5.loc[t(44):t(48)],
            'df2_6_down_0' : df2_6.loc[t(34):t(44)],
            'df2_6_up_0' : df2_6.loc[t(44):t(48)],
            'df2_6_down_1' : df2_6.loc[t(48):t(55)],
            'df2_6_up_1' : df2_6.loc[t(55):t(60.)],
            'df2_6_down_2' : df2_6.loc[t(60.5):],
            'df2_7_down_0' : df2_7.loc[t(34):t(44)],
            'df2_7_up_0' : df2_7.loc[t(44):t(48)],
            'df2_7_down_1' : df2_7.loc[t(48):t(55)],
            'df2_7_up_1' : df2_7.loc[t(55):t(60.5)],
            'df2_7_down_2' : df2_7.loc[t(60.5):],
            'df2_8_down_0' : df2_8.loc[t(34):t(44)],
            'df2_8_up_0' : df2_8.loc[t(44):t(48)],
            'df2_8_down_1' : df2_8.loc[t(48):t(55)],
            'df2_8_up_1' : df2_8.loc[t(55):t(60.5)],
            'df2_8_down_2' : df2_8.loc[t(60.5):],
            'df2_9_down_0' : df2_9.loc[t(16):t(29)],
            'df2_9_up_0' : df2_9.loc[t(29):t(34)],
            'df2_9_down_1' : df2_9.loc[t(34):t(44)],
            'df2_9_up_1' : df2_9.loc[t(44):t(48)],
            'df2_9_down_2' : df2_9.loc[t(48):t(55)],
            'df2_9_up_2' : df2_9.loc[t(55):t(60.5)],
            'df2_9_down_3' : df2_9.loc[t(60.5):],
            'df3_0_up_0' : df3_0.loc[t(6):t(10)],
            'df3_0_down_1' : df3_0.loc[t(10):t(26)],
            'df3_1_down_0' : df3_1.loc[:t(6)],
            'df3_1_up_0' : df3_1.loc[t(6):t(10)],
            'df3_1_down_1' : df3_1.loc[t(10):t(26)],
            'df3_1_up_1' : df3_1.loc[t(26):t(29)],
            'df3_2_down_0' : df3_2.loc[:t(6)],
            'df3_2_up_0' : df3_2.loc[t(6):t(10)],
            'df3_2_down_1' : df3_2.loc[t(10):t(26)],
            'df3_2_up_1' : df3_2.loc[t(26):t(29)],
            'df3_2_down_2' : df3_2.loc[t(29):t(46)],
            'df3_3_down_0' : df3_3.loc[:t(6)],
            'df3_3_up_0' : df3_3.loc[t(6):t(10)],
            'df3_3_down_1' : df3_3.loc[t(10):t(26)],
            'df4_0_down_0' : df4_0.loc[:t(4)],
            'df4_0_up_0' : df4_0.loc[t(4):t(17)],
            'df4_0_down_1' : df4_0.loc[t(17):t(31)],
            'df4_0_up_1' : df4_0.loc[t(31):t(41)],
            'df4_0_down_2' : df4_0.loc[t(41):t(52)],
            'df4_0_up_2' : df4_0.loc[t(52):t(60)],
            'df4_0_down_3' : df4_0.loc[t(60):t(71)],
            'df4_0_up_3' : df4_0.loc[t(71):t(79)],
            'df5_0_down_0' : df5_0.loc[t(56):t(65)],
            'df5_0_up_0' : df5_0.loc[t(65):t(70)],
            'df5_0_down_1' : df5_0.loc[t(70):],
            'df5_1_up_0' : df5_1.loc[t(5):t(11)],
            'df5_1_down_1' : df5_1.loc[t(11):t(17)],
            'df5_2_down_0' : df5_2.loc[t(56):t(65)],
            'df5_2_up_0' : df5_2.loc[t(65):t(70)],
            'df5_3_down_0' : df5_3.loc[:t(5)],
            'df5_3_up_0' : df5_3.loc[t(5):t(11)],
            'df5_4_down_0' : df5_4.loc[t(41):t(50)],
            'df5_4_up_0' : df5_4.loc[t(50):t(56)],
            'df5_5_down_0' : df5_5.loc[:t(5)],
            'df5_5_up_0' : df5_5.loc[t(5):t(11)],
            'df5_5_down_1' : df5_5.loc[t(11):t(17)],
            'df5_5_up_1' : df5_5.loc[t(17):t(20)],
            'df5_5_down_2' : df5_5.loc[t(20):t(35)],
            'df5_6_down_0' : df5_6.loc[t(20):t(35)],
            'df5_6_up_0' : df5_6.loc[t(35):t(41)],
            'df5_6_down_1' : df5_6.loc[t(41):t(50)],
            'df5_7_down_0' : df5_7.loc[:t(5)],
            'df5_7_up_0' : df5_7.loc[t(5):t(11)],
            'df5_7_down_1' : df5_7.loc[t(11):t(17)],
            'df5_7_up_1' : df5_7.loc[t(17):t(20)],
            'df5_7_down_2' : df5_7.loc[t(20):t(35)],
            'df5_7_up_2' : df5_7.loc[t(35):t(41)],
            'df5_8_down_0' : df5_8.loc[:t(5)],
            'df5_8_up_0' : df5_8.loc[t(5):t(11)],
            'df5_8_down_1' : df5_8.loc[t(11):t(17)],
            'df5_8_up_1' : df5_8.loc[t(17):t(20)],
            'df5_8_down_2' : df5_8.loc[t(20):t(35)],
            'df5_8_up_2' : df5_8.loc[t(35):t(41)],
            'df5_8_down_3' : df5_8.loc[t(41):t(50)],
            'df5_8_up_3' : df5_8.loc[t(50):t(56)],
            'df5_8_down_4' : df5_8.loc[t(56):t(64)],
            'df5_8_up_4' : df5_8.loc[t(64):t(70)],
            'df6_0_down_0' : df6_0.loc[t(50):t(65)],
            'df6_0_up_0' : df6_0.loc[t(65):t(68.5)],
            'df6_0_down_1' : df6_0.loc[t(68.5):],
            'df6_1_down_0' : df6_1.loc[:t(5)],
            'df6_1_up_0' : df6_1.loc[t(5):t(11)],
            'df6_1_down_1' : df6_1.loc[t(11):t(30.5)],
            'df6_1_up_1' : df6_1.loc[t(30.5):t(33)],
            'df6_1_down_2' : df6_1.loc[t(33):t(47)],
            'df6_1_up_2' : df6_1.loc[t(47):t(50)],
            'df6_1_down_3' : df6_1.loc[t(50):t(65)],
            'df6_2_down_0' : df6_2.loc[:t(5)],
            'df6_2_up_0' : df6_2.loc[t(5):t(11)],
            'df6_2_down_1' : df6_2.loc[t(11):t(30.5)],
            'df6_2_up_1' : df6_2.loc[t(30.5):t(33)],
            'df6_2_down_2' : df6_2.loc[t(33):t(47)],
            'df6_3_down_0' : df6_3.loc[t(11):t(30.5)],
            'df6_3_up_0' : df6_3.loc[t(30.5):t(33)],
            'df6_3_down_1' : df6_3.loc[t(33):t(47)],
            'df6_3_up_1' : df6_3.loc[t(47):t(50)],
            'df6_3_down_2' : df6_3.loc[t(50):t(65)],
            'df6_3_up_2' : df6_3.loc[t(65):t(68.5)],
            'df6_3_down_3' : df6_3.loc[t(68.5):],
            'df6_4_down_0' : df6_4.loc[t(11):t(30.5)],
            'df6_4_up_0' : df6_4.loc[t(30.5):t(33)],
            'df6_4_down_1' : df6_4.loc[t(33):t(47)],
            'df6_4_up_1' : df6_4.loc[t(47):t(50)],
            'df6_4_down_2' : df6_4.loc[t(50):t(65)],
            'df6_4_up_2' : df6_4.loc[t(65):t(68.5)],
            'df6_4_down_3' : df6_4.loc[t(68.5):],
            'df6_5_down_0' : df6_5.loc[t(11):t(30.5)],
            'df6_5_up_0' : df6_5.loc[t(30.5):t(33)],
            'df6_5_down_1' : df6_5.loc[t(33):t(47)],
            'df6_5_up_1' : df6_5.loc[t(47):t(50)],
            'df6_5_down_2' : df6_5.loc[t(50):t(65)],
            'df6_5_up_2' : df6_5.loc[t(65):t(68.5)],
            'df6_5_down_3' : df6_5.loc[t(68.5):],
            'df6_6_down_0' : df6_6.loc[t(33):t(47)],
            'df6_6_up_0' : df6_6.loc[t(47):t(50)],
            'df6_6_down_1' : df6_6.loc[t(50):t(65)],
            'df6_7_down_0' : df6_7.loc[t(33):t(47)],
            'df6_7_up_0' : df6_7.loc[t(47):t(50)],
            'df6_7_down_1' : df6_7.loc[t(50):t(65)],
            'df6_7_up_1' : df6_7.loc[t(65):t(68.5)],
            'df6_7_down_2' : df6_7.loc[t(68.5):],
            'df6_8_down_0' : df6_8.loc[t(33):t(47)],
            'df6_8_up_0' : df6_8.loc[t(47):t(50)],
            'df6_8_down_1' : df6_8.loc[t(50):t(65)],
            'df6_8_up_1' : df6_8.loc[t(65):t(68.5)],
            'df6_9_down_0' : df6_9.loc[t(50):t(65)],
            'df6_9_up_0' : df6_9.loc[t(65):t(68.5)],
            'df6_9_down_1' : df6_9.loc[t(68.5):],
            'df6_10_down_0' : df6_10.loc[t(50):t(65)],
            'df6_10_up_0' : df6_10.loc[t(65):t(68.5)],
            'df6_10_down_1' : df6_10.loc[t(68.5):],
            'df15_0_down_0' : df15_0.loc[:t(4)],
            'df15_0_up_0' : df15_0.loc[t(4):t(9)],
            'df15_0_down_1' : df15_0.loc[t(9):t(19)],
            'df15_0_up_1' : df15_0.loc[t(19):t(27)],
            'df15_0_down_2' : df15_0.loc[t(27):],
            'df15_1_down_0' : df15_1.loc[:t(4)],
            'df15_1_up_0' : df15_1.loc[t(4):t(9)],
            'df15_1_down_1' : df15_1.loc[t(9):t(19)],
            'df15_1_up_1' : df15_1.loc[t(19):t(27)],
            'df15_1_down_2' : df15_1.loc[t(27):],
            'df15_2_down_0' : df15_2.loc[:t(4)],
            'df15_2_up_0' : df15_2.loc[t(4):t(9)],
            'df15_2_down_1' : df15_2.iloc[t(9):t(19)],
            'df15_2_up_1' : df15_2.loc[t(19):t(27)],
            'df15_2_down_2' : df15_2.loc[t(27):],
            'df16_0_down_0' : df16_0.loc[:t(6)],
            'df16_0_up_0' : df16_0.loc[t(6):t(21)],
            'df16_0_down_1' : df16_0.loc[t(21):t(34)], # Misaligning data
            'df16_0_up_1' : df16_0.loc[t(34):t(41)],
            'df16_0_down_2' : df16_0.loc[t(41):],
            'df17_0_down_0' : df17_0.loc[:t(6.2)],
            'df17_0_up_0' : df17_0.loc[t(6.2):t(12)],
            'df17_0_down_1' : df17_0.loc[t(12):t(27.2)],
            'df17_0_up_1' : df17_0.loc[t(27.2):t(33.5)],
            'df17_0_down_2' : df17_0.loc[t(34):t(45.5)],
            'df17_0_up_2' : df17_0.loc[t(45.5):t(48)],
            'df17_0_down_3' : df17_0.loc[t(48):],
            'df17_1_down_0' : df17_1.loc[:t(6.2)],
            'df17_1_up_0' : df17_1.loc[t(6.2):t(12)],
            'df17_1_down_1' : df17_1.loc[t(12):t(27.2)],
            'df17_2_down_0' : df17_2.loc[:t(6.2)],
            'df17_2_up_0' : df17_2.loc[t(6.2):t(12)],
            'df17_2_down_1' : df17_2.loc[t(12):t(27.2)],
            'df17_2_up_1' : df17_2.loc[t(27.2):t(33.5)],
            'df17_2_down_2' : df17_2.loc[t(34):t(45.5)],
            'df17_3_down_0' : df17_3.loc[:t(6.2)],
            'df17_3_up_0' : df17_3.loc[t(6.2):t(12)],
            'df17_3_down_1' : df17_3.loc[t(12):t(27.2)],
            'df17_3_up_1' : df17_3.loc[t(27.2):t(33.5)],
            'df17_4_up_0' : df17_4.loc[t(6.2):t(12)],
            'df17_4_down_1' : df17_4.loc[t(12):t(27.2)],
            'df17_4_up_1' : df17_4.loc[t(27.2):t(33.5)],
            'df17_4_down_2' : df17_4.loc[t(34):t(45.5)],
            'df17_4_up_2' : df17_4.loc[t(45.5):t(48)],
            'df17_4_down_3' : df17_4.loc[t(48):],
            'df17_5_down_0' : df17_0.loc[:t(6.2)],
            'df17_5_up_0' : df17_0.loc[t(6.2):t(12)],
            'df17_5_down_1' : df17_0.loc[t(12):t(27.2)],
            'df17_5_up_1' : df17_0.loc[t(27.2):t(33.5)],
            'df17_5_down_2' : df17_0.loc[t(34):t(45.5)],
            'df17_5_up_2' : df17_0.loc[t(45.5):t(48)],
            'df17_5_down_3' : df17_0.loc[t(48):],
            'df17_6_down_0' : df17_0.loc[:t(6.2)],
            'df17_6_up_0' : df17_0.loc[t(6.2):t(12)],
            'df17_6_down_1' : df17_0.loc[t(12):t(27.2)], # Misaligning data
            'df17_6_up_1' : df17_0.loc[t(27.2):t(33.5)],
            'df17_6_down_2' : df17_0.loc[t(34):t(45.5)],
            'df17_6_up_2' : df17_0.loc[t(45.5):t(48)],
            'df17_6_down_3' : df17_0.loc[t(48):],
            'df18_0_down_0' : df18_0.loc[:t(7.8)],
            'df18_0_up_0' : df18_0.loc[t(7.8):t(11.5)],
            'df18_0_down_1' : df18_0.loc[t(11.5):t(24)],
            'df18_0_up_1' : df18_0.loc[t(24):t(30)],
            'df18_1_down_0' : df18_1.loc[:t(7.8)],
            'df18_1_up_0' : df18_1.loc[t(7.8):t(11.5)],
            'df18_1_down_1' : df18_1.loc[t(11.5):t(24)],
            'df18_1_up_1' : df18_1.loc[t(24):t(30)],
            'df18_1_down_2' : df18_1.loc[t(30):t(38)],
            'df18_1_up_2' : df18_1.loc[t(38):t(45.5)],
            'df18_1_down_3' : df18_1.loc[t(45.5):],
            'df18_2_down_0' : df18_2.loc[:t(7.8)],
            'df18_2_up_0' : df18_2.loc[t(7.8):t(11.5)],
            'df18_2_down_1' : df18_2.loc[t(11.5):t(24)],
            'df18_2_up_1' : df18_2.loc[t(24):t(30)],
            'df18_2_down_2' : df18_2.loc[t(30):t(38)],
            'df18_2_up_2' : df18_2.loc[t(38):t(45.5)],
            'df18_2_down_3' : df18_2.loc[t(45.5):],
            'df18_3_down_0' : df18_3.loc[:t(7.8)],
            'df18_3_up_0' : df18_3.loc[t(7.8):t(11.5)],
            'df18_3_down_1' : df18_3.loc[t(11.5):t(24)],
            'df18_3_up_1' : df18_3.loc[t(24):t(30)],
            'df18_3_down_2' : df18_3.loc[t(30):t(38)],
            'df18_3_up_2' : df18_3.loc[t(38):t(45.5)],
            'df18_3_down_3' : df18_3.loc[t(45.5):],
            'df18_4_down_0' : df18_4.loc[:t(7.8)],
            'df18_4_up_0' : df18_4.loc[t(7.8):t(11.5)],
            'df18_4_down_1' : df18_4.loc[t(11.5):t(24)],
            'df18_5_down_0' : df18_5.loc[:t(7.8)],
            'df18_5_up_0' : df18_5.loc[t(7.8):t(11.5)],
            'df18_5_down_1' : df18_5.loc[t(11.5):t(24)],
            'df18_6_down_0' : df18_6.loc[:t(7.8)],
            'df18_6_up_0' : df18_6.loc[t(7.8):t(11.5)],
            'df18_6_down_1' : df18_6.loc[t(11.5):t(24)],
            'df18_6_up_1' : df18_6.loc[t(24):t(30)],
            'df18_6_down_2' : df18_6.loc[t(30):t(38)],
            'df18_6_up_2' : df18_6.loc[t(38):t(45.5)],
            'df18_7_up_0' : df18_7.loc[t(7.8):t(11.5)],
            'df18_7_down_1' : df18_7.loc[t(11.5):t(24)],
            'df18_7_up_1' : df18_7.loc[t(24):t(30)],
            'df18_8_up_0' : df18_8.loc[t(7.8):t(11.5)],
            'df18_8_down_1' : df18_8.loc[t(11.5):t(24)],
            'df18_8_up_1' : df18_8.loc[t(24):t(30)],
            'df18_8_down_2' : df18_8.loc[t(30):t(38)],
            'df18_9_up_0' : df18_9.loc[t(7.8):t(11.5)],
            'df18_9_down_1' : df18_9.loc[t(11.5):t(24)],
            'df18_9_up_1' : df18_9.loc[t(24):t(30)],
            'df19_0_up_0' : df19_0.loc[t(2):t(6.8)],
            'df19_0_down_1' : df19_0.loc[t(6.8):t(25.6)],
            'df19_0_up_1' : df19_0.loc[t(25.6):t(34.2)],
            'df19_0_down_2' : df19_0.loc[t(34.2):t(43)],
            'df19_1_up_0' : df19_1.loc[t(2):t(6.8)],
            'df19_1_down_1' : df19_1.loc[t(6.8):t(25.6)],
            'df19_1_up_1' : df19_1.loc[t(25.6):t(34.2)],
            'df19_1_down_2' : df19_1.loc[t(34.2):t(43)],
            'df19_1_up_2' : df19_1.loc[t(43):t(46)],
            'df19_1_down_3' : df19_1.loc[t(46):],
            'df19_2_up_0' : df19_2.loc[t(2):t(6.8)],
            'df19_2_down_1' : df19_2.loc[t(6.8):t(25.6)],
            'df19_3_up_0' : df19_3.loc[t(2):t(6.8)],
            'df19_3_down_1' : df19_3.loc[t(6.8):t(25.6)],
            'df19_3_up_1' : df19_3.loc[t(25.6):t(34.2)],
            'df19_3_down_2' : df19_3.loc[t(34.2):t(43)],
            'df19_3_up_2' : df19_3.loc[t(43):t(46)],
            'df19_3_down_3' : df19_3.loc[t(46):],
            'df19_4_up_0' : df19_4.loc[t(2):t(6.8)],
            'df19_4_down_1' : df19_4.loc[t(6.8):t(25.6)],
            'df19_6_down_1' : df19_6.loc[t(6.8):t(25.6)],
            'df19_6_up_1' : df19_6.loc[t(25.6):t(34.2)],
            'df19_6_down_2' : df19_6.loc[t(34.2):t(43)],
            'df19_6_up_2' : df19_6.loc[t(43):t(46)],
            'df19_6_down_3' : df19_6.loc[t(46):],
            'df19_7_down_1' : df19_7.loc[t(6.8):t(25.6)],
            'df19_7_up_1' : df19_7.loc[t(25.6):t(34.2)],
            'df19_7_down_2' : df19_7.loc[t(34.2):t(43)],
            'df19_7_up_2' : df19_7.loc[t(43):t(46)],
            'df19_7_down_3' : df19_7.loc[t(46):],
            'df19_8_down_1' : df19_8.loc[t(6.8):t(25.6)],
            'df19_8_up_1' : df19_8.loc[t(25.6):t(34.2)],
            'df19_8_down_2' : df19_8.loc[t(34.2):t(43)],
            'df19_8_up_2' : df19_8.loc[t(43):t(46)],
            'df19_8_down_3' : df19_8.loc[t(46):],
            'df19_9_up_1' : df19_9.loc[t(25.6):t(34.2)],
            'df19_9_down_2' : df19_9.loc[t(34.2):t(43)],
            'df19_9_up_2' : df19_9.loc[t(43):t(46)],
            'df19_9_down_3' : df19_9.loc[t(46):],
            'df19_10_up_1' : df19_10.loc[t(25.6):t(34.2)],
            'df19_10_down_2' : df19_10.loc[t(34.2):t(43)],
            'df19_10_up_2' : df19_10.loc[t(43):t(46)]
        }
