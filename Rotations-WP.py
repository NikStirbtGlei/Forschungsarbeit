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
fluid='Argon'
r_max= 0.5
r_min=0.1
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
r_span=(r_min, r_max)
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

#Zustand des Fluids updaten auf p2(r_max) und T2(r_max)
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
T_Vorlauf = 273.15 + 53
T_Nachlauf = 273.15 + 35
T2 = T_Vorlauf + 3
T3 = T_Nachlauf +3

# Fehlermeldung falls Teperatur nicht im Bereich von r erfasst wird
if not np.nanmin(T) <= T2 <= np.nanmax(T):
    raise ValueError('Temperatur liegt nicht im Radiusbereich')

#Ausgabe von r bei T(r)= T2 und von p(r2)
r2 = np.interp(T2, T, r)
p2 = np.interp(r2, r, p)
p3 = p2

print(f'Für T2 = {T2}K ist r = {r2}m und p = {p2} Pa ')

#Integrationsintervall definieren
T_span = np.linspace(T2, T3, 300)

#cp(p,T) bekommen als Array über die Temperaturen T und dem Druck p3
cp = []
for T in T_span:
    AS.update(CP.PT_INPUTS, p2, T)
    cp.append(AS.cpmass())

#Zustand updaten
AS.update(CP.PT_INPUTS, p3, T3)
#Integration cp(p,T) über T
delta_h_23 = np.trapezoid(cp, T_span)
m_flow = Q_dot / delta_h_23

print(f'{fluid}-Masstestrom beträgt {m_flow} kg/s um einen Wärmestrom von {Q_dot}W zu generieren')

