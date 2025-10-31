# Packages
import numpy as np
import pandas as pd
from pandas.core.nanops import nanmin
from scipy.integrate import solve_ivp
from CoolProp import CoolProp as CP
from CoolProp.CoolProp import AbstractState, PropsSI
import matplotlib.pyplot as plt

import pytest #Test-Framework
import matplotlib.pyplot as plt #Visualisierung Graphen usw.

#Zustandspunkt 1
T1 = 268.15
p1 = 300000
fluid='Krypton'
r2= 0.5
r1=0.1
schritte=200
n=5000/60 # Umdrehungen pro s [1/s]
omega=n*2*np.pi

#Argon als Fluid und Zustand definieren

AS= AbstractState('HEOS',fluid)  #Objekt für Argon mit Zugriff auf realen Stoffdaten
AS.update(CP.PT_INPUTS, p1, T1)  #Updatet Objekt auf Zustand mit p1, T1
s12_const=AS.smass()                   #spezifische Entropie bei p1 , T1

#Differentialgleichung für Druck

def dp(r,p):
    AS.update(CP.PSmass_INPUTS, float(p[0]), s12_const) #Zustand setzen mit p und s12_const
    rho = AS.rhomass()                              #Dichte bei Zustand p, s12_const
    return rho*(omega**2)*r                         # aus DGL: dp/dr= rho(p,T)*omega^2 * r

#Integrationsintervall deffinieren
r_span=(r1, r2)
r_schritte = np.linspace(r_min,r_max,schritte) #Erzeugt Array von r_min bis r_max mit Schritte

#DGL lösen Integrator
sol=solve_ivp(dp,r_span,[p1],t_eval=r_schritte,rtol=1e-7, atol=1e-9)
if not sol.success:
    print('Shit, war nix:', sol.message)
if sol.success:
    print('Bombe! hat geklappt')

#Ergebnisse als Arrays
r= sol.t
p= sol.y[0]
#p2 bei r_max setzen
p_max= p[-1]

#Zustand des Fluids updaten auf p2_max und T2(r_max)
AS.update(CP.PSmass_INPUTS, p_max, s12_const)
T2 = AS.T()

#Leeres Array erstellen mit den Dimensionen von r und p
T = np.empty_like(p)

#Basierend auf Ergebnisse von p, mit s1 T berechnen
for i, pi in enumerate(p):
    AS.update(CP.PSmass_INPUTS, float(pi), s12_const)
    T[i] = AS.T()

# --- Druck und Temperatur gemeinsam plotten ---
fig, ax1 = plt.subplots()

# Linke Achse (Druck)
ax1.plot(r, p/1e5, color='tab:blue', label='Druck [bar]')
ax1.set_xlabel('Radius r [m]')
ax1.set_ylabel('Druck p [bar]', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.grid(True)

# Rechte Achse (Temperatur)
ax2 = ax1.twinx()  # zweite y-Achse erstellen
ax2.plot(r, T, color='tab:orange', linestyle='--', label='Temperatur [K]')
ax2.set_ylabel('Temperatur T [K]', color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')

# Gemeinsamer Titel und Layout
plt.title(f'Isentroper Verlauf: Druck & Temperatur {fluid}')
fig.tight_layout()
plt.show()

#Solltemperaturen und Wärmestrom setzen
Q_dot=-6000 #Q_dot [W]
T_Nachlauf = 273.15 + 35
T3 = T_Nachlauf +3
p2 = float(p_max)
p3 = p2

# delta h23 berechnen
AS.update(CP.PT_INPUTS, p2, T2)
h2 = AS.hmass()
AS.update(CP.PT_INPUTS, p3, T3)
h3 = AS.hmass()
delta_h_23 = h3 - h2

#Massestrom bestimmen
m_flow = Q_dot / delta_h_23

#Entropie von Zustand 3 und 4 bestimmen (isentroper Fall)
s34_const = AS.smass()

#p4 berechnen - muss kleiner als p1 sein da p4=p5 und p5<p1
#1. Temperaturhub delta_T12 deffinieren, davon 10% für delta_T_51
delta_T_12 = T2-T1
delta_T_51 = delta_T_12/10
#2. T5 bestimmen
T5 = T1 - delta_T_51

#3 Druckniveau bestimmen bei T = T5 und s= s12_const. --> p5 = p4 (isobare Wärmezufuhr)

AS.update(CP.SmassT_INPUTS, s12_const, T5)
p5 = AS.p()
p4 = p5

#T4 bestimmen
AS.update(CP.PSmass_INPUTS, p4, s34_const)
T4 = AS.T()

# Q_dot_in bestimmen für Wärmeaufnahme
h4 = AS.hmass()
AS.update(CP.PT_INPUTS, p5, T5)
h5 = AS.hmass()
delta_h_45 = h5 - h4
Q_dot_in = m_flow * delta_h_45

# Schritt 5-->1 wieder isentrope Verdichtung durch Lüfter
AS.update(CP.PT_INPUTS, p1, T1)
h1 = AS.hmass()

delta_h_51= h1 - h5

P_Luefter = m_flow * delta_h_51

#P_Verdichtung bestimmen
delta_h_12 = h2 - h1
P_Verdichtung = m_flow * delta_h_12

#P_Expansion bestimmen
delta_h_34 = h4 - h3
P_Expansion =  m_flow * delta_h_34

# Temperaturdeltas bestimmen

delta_T_34 = T3 - T4

Erster_Hauptsatz = Q_dot + Q_dot_in + P_Luefter + P_Verdichtung + P_Expansion

print(f'[T1,p1]={T1,p1} , [T2,p2]={T2,p2} , [T3,p3]={T3,p3} , [T4,p4]={T4,p4} , [T5,p5]={T5,p5} , Q_dot = {Q_dot} , Q_dot_in = {Q_dot_in}, m_flow = {m_flow} , P_Luefter ={P_Luefter} , P_Verdichtung = {P_Verdichtung} , P_Expansion = {P_Expansion}')
print(f'Erster Hauptsatz = {Erster_Hauptsatz}')
print(f'delta_T_12=', {delta_T_12})
print(f'delta_T_34=', {delta_T_34})

# Diagramme

# ----- Punkte definieren (nur bereits berechnete Zustände verwenden) -----
points = [
    ("1", p1, T1),
    ("2", p2, T2),
    ("3", p3, T3),
    ("4", p4, T4),
    ("5", p5, T5),
]
# Kreis schließen
points_closed = points + [points[0]]

# Zustandsgrößen aus CoolProp holen
s_list, T_list, h_list, p_list, labels = [], [], [], [], []
for lab, p_i, T_i in points_closed:
    AS.update(CP.PT_INPUTS, float(p_i), float(T_i))
    labels.append(lab)
    s_list.append(AS.smass())   # J/(kg·K)
    T_list.append(AS.T())       # K
    h_list.append(AS.hmass())   # J/kg
    p_list.append(AS.p())       # Pa

s = np.array(s_list)
T = np.array(T_list)
h = np.array(h_list) / 1e3      # -> kJ/kg
p = np.array(p_list) / 1e5      # -> bar

# ----- T-s Diagramm (Punkte + Verbindungs­linien) -----
plt.figure()
plt.plot(s, T, "-o")
for i, lab in enumerate(labels):
    if i < len(points):  # letztes = geschlossenes 1 nicht nochmal beschriften
        plt.text(s[i], T[i], f"  {lab}")
plt.xlabel("s [J/(kg·K)]")
plt.ylabel("T [K]")
plt.title(f"Kreisprozess {fluid}: T–s (nur Punkte & Linien)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ----- p-h Diagramm (Punkte + Verbindungs­linien) -----
plt.figure()
plt.semilogy(h, p, "-o")
for i, lab in enumerate(labels):
    if i < len(points):
        plt.text(h[i], p[i], f"  {lab}")
plt.xlabel("h [kJ/kg]")
plt.ylabel("p [bar] (log)")
plt.title(f"Kreisprozess {fluid}: p–h (nur Punkte & Linien)")
plt.grid(True, which="both")
plt.tight_layout()
plt.show()


# Leistungen überprüfen , plausibilisieren, nicht T2 vorgeben sondern r2 und n --> T2 als ergebnis
# Qellenseitigen externen Strom Wasser/Wasser Sole/Wasser
# nächsten Schritte: Verdichtungs und Expansionsgütegrad einbringen + Lüfter wirkungsgrad