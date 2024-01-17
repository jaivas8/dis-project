import matplotlib.pyplot as plt
import numpy as np

def bezier_curve(t, p0, p1, p2, p3):
    return (1-t)**3*p0 + 3*(1-t)**2*t*p1 + 3*(1-t)*t**2*p2 + t**3*p3

def generate_smooth_curve(point1, point2, control_points):
    t = np.linspace(0, 1, 100)
    x = bezier_curve(t, point1[0], control_points[0][0], control_points[1][0], point2[0])
    y = bezier_curve(t, point1[1], control_points[0][1], control_points[1][1], point2[1])
    print(x)
    print(y)
    return x, y

point1 = (-0.5, -0.5)
point2 = (-0.9, -0.9)

# Control points for the Bézier curve
control_point1 = (1, 1)
control_point2 = (-1, -1)

# Generate the smooth curve
curve_x, curve_y = generate_smooth_curve(point1, point2, [control_point1, control_point2])

# Plotting
plt.plot(curve_x, curve_y, label='Smooth Curve')
plt.scatter([point1[0], point2[0]], [point1[1], point2[1]], color='red', label='End Points')
plt.scatter([control_point1[0], control_point2[0]], [control_point1[1], control_point2[1]], color='green', label='Control Points')
plt.legend()
plt.show()
