# Packages
import numpy as np
import pandas as pd
from pandas.core.nanops import nanmin
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
from CoolProp import CoolProp as CP
from CoolProp.CoolProp import AbstractState, PropsSI
import matplotlib.pyplot as plt #Visualisierung Graphen usw.
from scipy.interpolate import interp1d

#Zustandspunkt 1
T1 = 268.15
p1 = 300000
fluid='Krypton'
r2= 0.5
r1=0.2
r0=0.08
n=5000/60 # Umdrehungen pro s [1/s]
omega=n*2*np.pi
eta_Verdichtung = 0.99
eta_Luefter = 0.8
eta_Expansion = 0.99

#Argon als Fluid und Zustand definieren
AS= AbstractState('HEOS',fluid)  #Objekt für Argon mit Zugriff auf realen Stoffdaten

# Zustand 1 setzen
AS.update(CP.PT_INPUTS, p1, T1)

#Anfangszustand s1 und h1
h1 = AS.hmass()
s1 = AS.smass()

#Differentialgleichung für für isentrope Druckerhöhung
AS_12_isentrop = AbstractState('HEOS',fluid)
r_eval = np.linspace(r1, r2, 500)

def dp_iso_12(r, p):
    pi = p[0]  # aktueller Druck
    AS_12_isentrop.update(CP.PSmass_INPUTS, pi, s1)
    rho_12_s = AS_12_isentrop.rhomass()
    return rho_12_s * omega**2 * r

#DGL lösen
sol_isentrop_12 = solve_ivp(dp_iso_12, (r1,r2), [p1],method="DOP853",t_eval=r_eval, rtol=1e-7, atol=1e-9)

# Differentialgleichung für reale Verdichtung
AS_12_isentrop_ref = AbstractState('HEOS',fluid)
AS_12_real = AbstractState('HEOS',fluid)
def dp_real_12(r,p):
    pi = p[0]
    AS_12_isentrop_ref.update(CP.PSmass_INPUTS, pi, s1) #Setzt Referenzzustand bei s=1 (isentrop)
    h_12_s = AS_12_isentrop_ref.hmass()
    h_real_12 = h1 + (h_12_s - h1) / eta_Verdichtung
    AS_12_real.update(CP.HmassP_INPUTS, h_real_12, pi)
    rho_12_real = AS_12_real.rhomass()
    return rho_12_real*omega**2 * r

#DGL lösen
sol_real_12 = solve_ivp(dp_real_12,(r1, r2),[p1], method="DOP853",t_eval = r_eval, rtol=1e-7, atol=1e-9)

#Lösung aus DGLs
r12= sol_isentrop_12.t
p_s_12_prof= sol_isentrop_12.y[0]
p_12_prof = sol_real_12.y[0]

# Temperatur, Enthalpie, Entropie und Druckprofile
T_s_12_prof = np.empty_like(r12)
T_12_prof = np.empty_like(r12)
s_12_prof = np.empty_like(r12)
h_s_12_prof = np.empty_like(r12)
h_12_prof = np.empty_like(r12)

# Isentroper Fall
for i, pi in enumerate(p_s_12_prof):
    AS_12_isentrop.update(CP.PSmass_INPUTS, pi, s1)
    h_s_12_prof[i] = AS_12_isentrop.hmass()
    T_s_12_prof[i] = AS_12_isentrop.T()

#Realer Fall
for i, pi in enumerate(p_12_prof):
    AS_12_isentrop_ref.update(CP.PSmass_INPUTS, pi, s1)
    h_s_12_ref = AS_12_isentrop_ref.hmass()
    h_12_prof[i] = h1 + (h_s_12_ref - h1) / eta_Verdichtung
    AS_12_real.update(CP.HmassP_INPUTS, h_12_prof[i], pi)
    T_12_prof[i] = AS_12_real.T()
    s_12_prof[i] = AS_12_real.smass()

#Zustandspunkte Real
p2 = p_12_prof[-1]
T2 = T_12_prof[-1]
h2 = h_12_prof[-1]
s2= s_12_prof[-1]

# Referenzzustand
p2_s = p_s_12_prof[-1]
T2_s = T_s_12_prof[-1]
h2_s = h_s_12_prof[-1]
s2_s = s1



#Solltemperaturen und Wärmestrom setzen
Q_dot_ab=-6000
T_Nachlauf = 273.15 + 35
T3 = T_Nachlauf +3
p3 = p2

# Fluid auf Zustand 3 updaten
AS_23 = AbstractState('HEOS',fluid)
AS_23.update(CP.PT_INPUTS, p3, T3)
s3 = AS_23.smass()

#Enthalpiedelta
h3 = AS_23.hmass()
delta_h_23 = h3 - h2

#Massestrom bestimmen
m_flow = Q_dot_ab / delta_h_23


# DGL Expansion isentrop
r_eval_expansion_34 = np.linspace(r2, r1, 500)
AS_34_isentrop = AbstractState('HEOS',fluid)
def dp_iso_34(r,p):
    pi = p[0]
    AS_34_isentrop.update(CP.PSmass_INPUTS, pi, s3)
    rho_34_s = AS_34_isentrop.rhomass()
    return rho_34_s * omega**2 * r

sol_isentrop_34 = solve_ivp(dp_iso_34,(r2, r1),[p3], method="DOP853",t_eval = r_eval_expansion_34, rtol=1e-7, atol=1e-9)

# DGL Expansion real
AS_34_real = AbstractState('HEOS',fluid)
AS_34_isentrop_ref = AbstractState('HEOS',fluid)

def dp_real_34(r,p):
    pi = p[0]
    AS_34_isentrop_ref.update(CP.PSmass_INPUTS, pi, s3)
    h_4_s = AS_34_isentrop_ref.hmass()
    h_4_real = h3 - eta_Expansion*(h3-h_4_s)
    AS_34_real.update(CP.HmassP_INPUTS, h_4_real, pi)
    rho_34_real = AS_34_real.rhomass()
    return rho_34_real * omega**2 * r

sol_real_34 = solve_ivp(dp_real_34,(r2, r1),[p3], method="DOP853",t_eval = r_eval_expansion_34, rtol=1e-7, atol=1e-9)

#Lösungen aus DGL's
r= sol_isentrop_34.t
p_s_34_prof= sol_isentrop_34.y[0]
p_34_prof = sol_real_34.y[0]

# Temperatur, Enthalpie, Entropie und Druckprofile
T_s_34_prof = np.empty_like(r)
T_34_prof = np.empty_like(r)
s_34_prof = np.empty_like(r)
h_s_34_prof = np.empty_like(r)
h_34_prof = np.empty_like(r)

# Isentroper Fall
for i, pi in enumerate(p_s_34_prof):
    AS_34_isentrop.update(CP.PSmass_INPUTS, pi, s3)
    h_s_34_prof[i] = AS_34_isentrop.hmass()
    T_s_34_prof[i] = AS_34_isentrop.T()

#Realer Fall
for i, pi in enumerate(p_34_prof):
    AS_34_isentrop_ref.update(CP.PSmass_INPUTS, pi, s3)
    h_s_34_ref = AS_34_isentrop_ref.hmass()
    h_34_prof[i] = h3 - eta_Expansion*(h3-h_s_34_ref)
    AS_34_real.update(CP.HmassP_INPUTS, h_34_prof[i], pi)
    T_34_prof[i] = AS_34_real.T()
    s_34_prof[i] = AS_34_real.smass()

#Zustandspunkte Real
p4 = p_34_prof[-1]
T4 = T_34_prof[-1]
h4 = h_34_prof[-1]
s4= s_34_prof[-1]

# Referenzzustand
p4_s = p_s_34_prof[-1]
T4_s = T_s_34_prof[-1]
h4_s = h_s_34_prof[-1]
s4_s = s3

# Drcuknivaeu Wärmeaufnahme
p5 = p4
r_eval_expansion= np.linspace(r1, r0, 200)
#Isentrope Verdichtung von 7 nach 1 zurückrechnen
AS_71_isentrop = AbstractState('HEOS',fluid)
def dp_iso_71(r,p):
    pi = p[0]
    AS_71_isentrop.update(CP.PSmass_INPUTS, pi, s1)
    rho_71_s = AS_71_isentrop.rhomass()
    return rho_71_s * omega**2 * r

#Solver aufstellen, integration von r1 nach r0 (wie expansion)
sol_iso_71 = solve_ivp(dp_iso_71,(r1, r0),[p1], method="DOP853",t_eval = r_eval_expansion, rtol=1e-7, atol=1e-9)

#Ergebnise/Profile
r_71= sol_iso_71.t
p_s_71_prof= sol_iso_71.y[0]
p7_s = p_s_71_prof[-1]

# T5 - Finder --> Kreisprozess schließen


def T5_Finder(T5):
    AS_5_Finder = AbstractState('HEOS', fluid)
    AS_56_isentrop = AbstractState('HEOS', fluid)
    AS_67_isentrop = AbstractState('HEOS', fluid)
    AS_67_real = AbstractState('HEOS', fluid)
    AS_5_Finder.update(CP.PT_INPUTS, p5, T5)
    s5 = AS_5_Finder.smass()
    h5 = AS_5_Finder.hmass()

    def dp_iso_56_Finder(r, p):
        pi = p[0]
        AS_56_isentrop.update(CP.PSmass_INPUTS, pi, s5)
        rho_56_s = AS_56_isentrop.rhomass()
        return rho_56_s * omega ** 2 * r

    # Solver/Integrator für 5-->6
    sol_iso_56 = solve_ivp(dp_iso_56_Finder, (r1, r0), [p5], method="DOP853", t_eval=r_eval_expansion, rtol=1e-7, atol=1e-9)

    #Ergebnis
    p6_s = sol_iso_56.y[0][-1]
    AS_56_isentrop.update(CP.PSmass_INPUTS, p6_s, s5)
    h6_s = AS_56_isentrop.hmass()

    # Lüfter 6-->7 und für 7 muss gelten s7 = s1
    AS_67_isentrop.update(CP.PSmass_INPUTS, p7_s, s5)
    h7_s = AS_67_isentrop.hmass()
    h7 = h6_s + (h7_s - h6_s)/eta_Luefter
    AS_67_real.update(CP.HmassP_INPUTS, h7, p7_s)
    s7 = AS_67_real.smass()
    #Endgültige Funktion für Nullstelle
    return s7 - s1

#Funktion macht folgendes: p6 wird ausgerechnet aus der weiteren expansion von 5 --> 6 , wird Zustand 1 genommen und von da aus Zustand 7 bestimmt
#Im nächsten Schritt wird die Funktion T5 finder definiert. diese nimmt den Druck p5, nimmt eine Temperatur T5 an definiert so den Zustand 5 im anschluss wird isotrop expandiert auf 6
# von 6-->7 wird dann erstmal isentrop verdichtet und daraus dann die reale verdichtung ausgerechnet mit h7. der Zustand wird definiert als mit p7 und h7 und daraus bekommt man s7
# Wenn bei dem genommenem T gilt : s7 = s1, dann wurde der richtige T5 gefunden der den Kreisprozess schließt

#T5 finden dass den spaß erfüllt
T5_min=T4
T5_max= T1
sol_T5 = root_scalar(T5_Finder, bracket=[T5_min, T5_max], method='brentq', xtol=1e-6)
T5= sol_T5.root
print(f'T5 = {T5}')


#Zustand 5 setzen
AS_5_real = AbstractState('HEOS',fluid)
AS_5_real.update(CP.PT_INPUTS, p5, T5)
h5= AS_5_real.hmass()
s5= AS_5_real.smass()

#Jetzt richtig 5-->6 isentrope entspannung r1 --> r0

# 4-->5 isobare Wärmezufuhr
Q_dot_zu = m_flow * (h5-h4)

# --- Druckverlauf über Radius ---
plt.figure(figsize=(7,5))
plt.plot(r, p_12_prof/1e5, label='1→2 real (Verdichtung)', color='blue', linewidth=1.8)
plt.plot(r, p_s_12_prof/1e5, label='1→2 isentrop (Verdichtung)', color='red', linestyle='--', linewidth=1.8)
plt.plot(r, p_s_34_prof/1e5, label='3→4 isentrop (Expansion)', color='green', linestyle='--', linewidth=1.8)
plt.plot(r, p_34_prof/1e5, label='3→4 real (Expansion)', color='black', linewidth=1.8)

plt.xlabel("Radius r [m]")
plt.ylabel("Druck p [bar]")
plt.title(f"{fluid}: Druckverlauf entlang Radius (Verdichtung & Expansion)")
plt.legend()
plt.grid(True, which='both', linestyle=':')
plt.tight_layout()
plt.show()

# --- Temperaturverlauf über Radius ---
plt.figure(figsize=(7,5))
plt.plot(r, T_12_prof, label='1→2 real (Verdichtung)', color='blue', linewidth=1.8)
plt.plot(r, T_s_12_prof, label='1→2 isentrop (Verdichtung)', color='red', linestyle='--', linewidth=1.8)
plt.plot(r, T_s_34_prof, label='3→4 isentrop (Expansion)', color='green', linestyle='--', linewidth=1.8)
plt.plot(r, T_34_prof, label='3→4 real (Expansion)', color='black', linewidth=1.8)

plt.xlabel("Radius r [m]")
plt.ylabel("Temperatur T [K]")
plt.title(f"{fluid}: Temperaturverlauf entlang Radius (Verdichtung & Expansion)")
plt.legend()
plt.grid(True, which='both', linestyle=':')
plt.tight_layout()
plt.show()


# Leistungen überprüfen , plausibilisieren, nicht T2 vorgeben sondern r2 und n --> T2 als ergebnis XXXXX
# Qellenseitigen externen Strom Wasser/Wasser Sole/Wasser
# nächsten Schritte: Verdichtungs und Expansionsgütegrad einbringen + Lüfter wirkungsgrad

