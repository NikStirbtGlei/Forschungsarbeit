import numpy as np
import matplotlib.pyplot as plt
from CoolProp import CoolProp as CP
from CoolProp.CoolProp import AbstractState
import RWP
from RWP import rho_67_prof

#Eingabe Rohrradius und berechnung der Querschnittsfläche
r_Rohr = 0.006
N_Rohr = 12
A_Rohr = N_Rohr * np.pi*(r_Rohr**2)

# Anzahl der Rohre


#Eingabe technische Rauhigkeit der Rohre
k_mm = 0.04
k = k_mm/1000

#Massestrom aus RWP Skript
m_flow = RWP.m_flow

# Rohrlänge x als Vektor
# Radien aus RWP
r0 = RWP.r0
r1 = RWP.r1
r2 = RWP.r2
delta_r12 = r2-r1
delta_r01 = r1-r0

# Rohrlängen 2-->3 und 4-->5
L_23 = 1
L_45 = 1

# Rohrlänge als x-Koordinate
x1 = 0
x2 = delta_r12
x3 = x2 + L_23
x4 = x3 + delta_r12
x5 = x4 + L_45
x6 = x5 + delta_r01
x7 = x6
x8 = x7 + delta_r01

#Dichten aus RWP Skript
rho_x_1 = RWP.rho_1
rho_x_2 = RWP.rho_2
rho_x_3 = RWP.rho_3
rho_x_4 = RWP.rho_4
rho_x_5 = RWP.rho_5
rho_x_6 = RWP.rho_6
rho_x_7 = RWP.rho_7
rho_x_8 = RWP.rho_1

#Dichten als Array zusammenfassen
rho = np.array([rho_x_1, rho_x_2, rho_x_3, rho_x_4, rho_x_5, rho_x_6, rho_x_7, rho_x_8])

#Volumenströme als Array aus Dichte-Array
V_dot = m_flow / rho

#Strömungsgeschwindigkeiten berechnen als Array
v_flow = V_dot/A_Rohr

#Lineare interpolation zwischen den einzelnen Zuständen
N=200

v_12 = np.linspace(v_flow[0], v_flow[1], N, False)
v_23 = np.linspace(v_flow[1], v_flow[2], N, False)
v_34 = np.linspace(v_flow[2], v_flow[3], N, False)
v_45 = np.linspace(v_flow[3], v_flow[4], N, False)
v_56 = np.linspace(v_flow[4], v_flow[5], N, False)
v_67 = np.linspace(v_flow[5], v_flow[6], N, False)
v_78 = np.linspace(v_flow[6], v_flow[7], N, False)

v_segmente = [v_12 , v_23 , v_34 , v_45 , v_56 , v_67 , v_78]

#Arrays für x definieren
x12 = np.linspace(x1, x2, N, False)
x23 = np.linspace(x2, x3, N, False)
x34 = np.linspace(x3, x4, N, False)
x45 = np.linspace(x4, x5, N, False)
x56 = np.linspace(x5, x6, N, False)
x67 = np.linspace(x6, x7, N, False)
x78 = np.linspace(x7, x8, N, False)

x_segmente = [x12, x23, x34, x45, x56, x67, x78]

#Arrays zusammenfassen zu einem
v_gesamt = np.concatenate(v_segmente)
x_gesamt = np.concatenate(x_segmente)

#Plot
plt.figure()
plt.plot(x_gesamt, v_gesamt)
plt.xlabel("x [m]")
plt.ylabel("v(x) [m/s]")
plt.grid(True)
# Knotenpunkte (x1..x8) und zugehörige v-Werte
x_knoten = np.array([x1, x2, x3, x4, x5, x6, x7, x8])
v_knoten = v_flow                 # das ist ja dein Array mit 8 Werten

# Schwarze Punkte an den Knickstellen
plt.scatter(x_knoten, v_knoten, s=40, c='k', zorder=3)

# Labels: 1–7 und für den letzten Punkt wieder 1
labels = ['1', '2', '3', '4', '5', '6', '7', '1']

x_knoten = np.array([x1, x2, x3, x4, x5, x6, x7, x8])
v_knoten = v_flow
labels   = ['1', '2', '3', '4', '5', '6', '7', '1']

# Offsets in x- und y-Richtung (gleiche Länge wie x_knoten)
dx = [-0.1,  -0.1,  -0.1,  -0.1,   -0.1,  0.1,  0.1,  -0.1]
dy = [0.0, 0.0, -0.01, 0.0,  0.01,  0.0, 0.0, 0.0]

for xi, vi, lab, dxi, dyi in zip(x_knoten, v_knoten, labels, dx, dy):
    plt.text(xi + dxi, vi + dyi, lab,
             ha='center', va='center', fontsize=10, color='k')


plt.show()

#Viskosität mu holen
AS_Viscosity = AbstractState('HEOS', RWP.fluid)
p_RWP = [RWP.p1, RWP.p2, RWP.p3, RWP.p4, RWP.p5, RWP.p6_s, RWP.p7]
T_RWP = [RWP.T1, RWP.T2, RWP.T3, RWP.T4, RWP.T5, RWP.T6s, RWP.T7]
mu = []
for p,T in zip(p_RWP, T_RWP):
    AS_Viscosity.update(CP.PT_INPUTS, p, T)
    mu.append(AS_Viscosity.viscosity())

rho_states = np.array([rho_x_1, rho_x_2, rho_x_3, rho_x_4, rho_x_5, rho_x_6, rho_x_7])
v_states = np.array([v_12[0], v_23[0], v_34[0], v_45[0], v_56[0], v_67[0], v_78[0]])

mu_RWP = np.array(mu)
Re = (rho_states * v_states * 2 * r_Rohr)/mu_RWP
Re_min = np.round(np.min(Re),0)
Re_max = np.round(np.max(Re),0)
print(Re_min, Re_max)

# Re, v^2 und rho über Streckenabschnitte gemittelt
# Re
Re_12_mittel = (Re[0] + Re[1])/2
Re_23_mittel = (Re[1] + Re[2])/2
Re_34_mittel = (Re[2] + Re[3])/2
Re_45_mittel = (Re[3] + Re[4])/2
Re_56_mittel = (Re[4] + Re[5])/2
Re_67_mittel = (Re[5] + Re[6])/2
Re_78_mittel = (Re[6] + Re[0])/2

Re_mittel = np.array([Re_12_mittel, Re_23_mittel, Re_34_mittel, Re_45_mittel, Re_56_mittel, Re_67_mittel, Re_78_mittel])

#v^2 gemittelt über Rohrstreckenabschnitte
v_square_12_mittel = (v_12[0]**2 + v_12[0]*v_12[1] + v_12[1]**2)/3
v_square_23_mittel = (v_23[0]**2 + v_23[0]*v_23[1] + v_23[1]**2)/3
v_square_34_mittel = (v_34[0]**2 + v_34[0]*v_34[1] + v_34[1]**2)/3
v_square_45_mittel = (v_45[0]**2 + v_45[0]*v_45[1] + v_45[1]**2)/3
v_square_56_mittel = (v_56[0]**2 + v_56[0]*v_56[1] + v_56[1]**2)/3
v_square_67_mittel = (v_67[0]**2 + v_67[0]*v_67[1] + v_67[1]**2)/3
v_square_78_mittel = (v_78[0]**2 + v_78[0]*v_78[1] + v_78[1]**2)/3

v_square_mittel = np.array([v_square_12_mittel, v_square_23_mittel, v_square_34_mittel, v_square_45_mittel, v_square_56_mittel, v_square_67_mittel, v_square_78_mittel])

# rho gemittelt über Rohrstreckenabschnitte
rho_12_mittel = (rho_states[0] + rho_states[1])/2
rho_23_mittel = (rho_states[1] + rho_states[2])/2
rho_34_mittel = (rho_states[2] + rho_states[3])/2
rho_45_mittel = (rho_states[3] + rho_states[4])/2
rho_56_mittel = (rho_states[4] + rho_states[5])/2
rho_67_mittel = (rho_states[5] + rho_states[6])/2
rho_78_mittel = (rho_states[6] + rho_states[0])/2

rho_mittel = np.array([rho_12_mittel, rho_23_mittel, rho_34_mittel, rho_45_mittel, rho_56_mittel, rho_67_mittel, rho_78_mittel])

#Rohrlängen als Arrays für Druckverlustberechnung
L = np.array([delta_r12, L_23, delta_r12, L_45, delta_r01, 0, delta_r01 ])
d_k = (2*r_Rohr)/k

#Arrrays definieren für Bedingungen der Rechenvorschrift und Ergebnisse
d = 2*r_Rohr
Re_k_d = Re_mittel * (k/d)
lambdas = np.zeros_like(Re_k_d)
# Einträge in Re_k_d abarbeiten und der rechenvorschrift zuweisen, und dann im Ergebnisvektor lamdas mit richtigem index abspeichern
for i,r in enumerate(Re_k_d):
    # technisch Glatte Rohre
    if r <= 65:
        if 2300 < Re_mittel[i] <= 1e5:
            lambdas[i] = 0.3164 / (Re_mittel[i]**0.25)
        elif 1e5 < Re_mittel[i] <= 5*1e6:
            lambdas[i] = 0.0032 + (0.221 / (Re_mittel[i]**0.237))
        else:
            raise ValueError('Keine passende Rechenvorschrift für Druckverlust')
    # voll ausgebildete turbulente Strömung
    elif 65 < r < 1300:
        if d/k <= 200:
            lambdas[i] = (1+(8/(Re_mittel[i]*k/d)))/((2*np.log10(3.71*d/k))**2)
        else:
            lambdas[i] = (1.8*np.log10((k/(10*d))+(7/Re_mittel[i])))**(-2)
    # technisch raue Rohre
    else:
        lambdas[i] = (2 * np.log10(d/k)+1.14)**(-2)


# p_Verlust als Array nach Rohrsegmenten für ein Rohr
p_Verlust = (lambdas * L * rho_mittel * v_square_mittel)/(4*r_Rohr)

#Zusammenaddierter Druckverlust aus einem Rohr
p_Verlust_ges = np.sum(p_Verlust)

print('p_Verlust=', p_Verlust_ges)

# Wärmeübertragung
# Fluidobjekt definieren
AS_Waerme = AbstractState('HEOS', RWP.fluid)
# Pr bekommen für Zustände 2, 3, 4, 5
p2345 = np.array([RWP.p2, RWP.p3, RWP.p4, RWP.p5])
T2345 = np.array([RWP.T2, RWP.T3, RWP.T4, RWP.T5])
Pr2345 = np.empty_like(p2345)
lambda_t = np.empty_like(p2345)

for i, (p, T) in enumerate(zip(p2345, T2345)):
    AS_Waerme.update(CP.PT_INPUTS, p, T)
    lambda_t[i] = AS_Waerme.conductivity()
    Pr2345[i] = AS_Waerme.Prandtl()

# Pr über Rohrabschnitte mitteln
Pr_23_mittel = (Pr2345[0] + Pr2345[1]) / 2
Pr_45_mittel = (Pr2345[2] + Pr2345[3]) / 2

# Wärmeleitfähigkeit lambda mitteln
lambda_t_23_mittel = (lambda_t[0] + lambda_t[1])/2
lambda_t_45_mittel = (lambda_t[2] + lambda_t[3])/2

# Größen für Nusselt
Re_Nu = np.array([Re_23_mittel , Re_45_mittel])
ksi = (1.8 * np.log10(Re_Nu) - 1.5)**(-2)
Pr_Nu = np.array([Pr_23_mittel, Pr_45_mittel])
lambda_Nu = np.array([lambda_t_23_mittel, lambda_t_45_mittel])

# Längen der Rohre als Array
L_Nu = np.array([L_23, L_45])

# f1 , f2 wird vernachlässigt da geringe delta - T über Rohrlänge
f1 = 1 + ((d/L_Nu)**(2/3))

#Nu
Nu = ((ksi/8) * Re_Nu * Pr_Nu)/ (1 + 12.7 * np.sqrt((ksi/8)) * ((Pr_Nu**(2/3))-1)) * f1

#alpha
alpha = ( Nu * lambda_Nu ) / d

#delta T
delta_T23 = T2345[1] - T2345[0]
delta_T45 = T2345[3] - T2345[2]
delta_T = np.array([delta_T23, delta_T45])

# nach A auflösen
Q_Einzel = np.array([RWP.Q_dot_ab/N_Rohr, RWP.Q_dot_zu/N_Rohr])
A = Q_Einzel / (alpha*delta_T)
print('alpha = ', alpha)
#benötigte Länge
L_soll = A / ( np.pi * d )
print('L_soll=', L_soll)
print('delta_T = ', delta_T)
print(T2345[0])
