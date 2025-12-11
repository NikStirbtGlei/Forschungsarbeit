# Packages
import numpy as np
import pandas as pd
from pandas.core.nanops import nanmin
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
from CoolProp import CoolProp as CP
from CoolProp.CoolProp import AbstractState, PropsSI
import matplotlib.pyplot as plt #Visualisierung Graphen usw.
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import interp1d

#Zustandspunkt 1
T1 = 280
p1 = 4000000
fluid='Argon'
r2= 0.6
r1=0.22
r0=0.05
n=4000/60 # Umdrehungen pro s [1/s]
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
rho_1 = AS.rhomass()

#Differentialgleichung für für isentrope Druckerhöhung
AS_12_isentrop = AbstractState('HEOS',fluid)
r_eval = np.linspace(r1, r2, 500)

def dp_iso_12(r, p):
    pi = p[0]  # aktueller Druck
    AS_12_isentrop.update(CP.PSmass_INPUTS, pi, s1)
    rho_12_s = AS_12_isentrop.rhomass()
    return rho_12_s * (omega**2) * r

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

# Temperatur, Enthalpie, Entropie, Dichte und Druckprofile
T_s_12_prof = np.empty_like(r12)
T_12_prof = np.empty_like(r12)
s_12_prof = np.empty_like(r12)
h_s_12_prof = np.empty_like(r12)
h_12_prof = np.empty_like(r12)
rho_12_prof = np.empty_like(r12)

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
    rho_12_prof[i] = AS_12_real.rhomass()

#Zustandspunkte Real
p2 = p_12_prof[-1]
T2 = T_12_prof[-1]
h2 = h_12_prof[-1]
s2= s_12_prof[-1]
rho_2 = rho_12_prof[-1]
cp_2 = AS_12_real.cpmass()

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

#Dichte rho_3
rho_3 = AS_23.rhomass()

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
r34= sol_isentrop_34.t
p_s_34_prof= sol_isentrop_34.y[0]
p_34_prof = sol_real_34.y[0]

# Temperatur, Enthalpie, Entropie und Druckprofile über Radius
T_s_34_prof = np.empty_like(r34)
T_34_prof = np.empty_like(r34)
s_34_prof = np.empty_like(r34)
h_s_34_prof = np.empty_like(r34)
h_34_prof = np.empty_like(r34)
rho_34_prof = np.empty_like(r34)

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
    rho_34_prof[i] = AS_34_real.rhomass()

#Zustandspunkte Real
p4 = p_34_prof[-1]
T4 = T_34_prof[-1]
h4 = h_34_prof[-1]
s4= s_34_prof[-1]
rho_4 = rho_34_prof[-1]

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

#Ergebnisse
r71= sol_iso_71.t
p_s_71_prof= sol_iso_71.y[0]
p7_s = p_s_71_prof[-1]

# T5 - Finder
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
    sol_iso_56_T5_Finder = solve_ivp(dp_iso_56_Finder, (r1, r0), [p5], method="DOP853", t_eval=r_eval_expansion, rtol=1e-7, atol=1e-9)

    #Ergebnis
    p6_s = sol_iso_56_T5_Finder.y[0][-1]
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

#Funktion macht folgendes: p6 wird ausgerechnet aus der weiteren expansion von 5 --> 6 und es wird Zustand 1 genommen und von da aus Zustand 7 bestimmt
#Im nächsten Schritt wird die Funktion T5 finder definiert. diese nimmt den Druck p5, nimmt eine Temperatur T5 an definiert so den Zustand 5 im anschluss wird isotrop expandiert auf das zuvor bestimmte p6
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
rho_5 = AS_5_real.rhomass()

#Jetzt richtig 5-->6 isentrope entspannung r1 --> r0 um Verlauf zu bekommen
AS_56_isentrop = AbstractState('HEOS', fluid)
def dp_iso_56(r, p):
    pi = p[0]
    AS_56_isentrop.update(CP.PSmass_INPUTS, pi, s5)
    rho_56_s = AS_56_isentrop.rhomass()
    return rho_56_s * omega ** 2 * r

# Solver/Integrator für 5-->6
sol_iso_56 = solve_ivp(dp_iso_56, (r1, r0), [p5], method="DOP853", t_eval=r_eval_expansion, rtol=1e-7, atol=1e-9)

#Ergebnisse (jetzt global gespeichert)
r56 = sol_iso_56.t
p_s_56_prof = sol_iso_56.y[0]

# Temperatur, Enthalpie, Entropie und Druckprofile über Radius Arrays erstellen
T_s_56_prof = np.empty_like(r56)
h_s_56_prof = np.empty_like(r56)
rho_s_56_prof = np.empty_like(r56)

# Isentroper Fall Arrays befüllen
for i, pi in enumerate(p_s_56_prof):
    AS_56_isentrop.update(CP.PSmass_INPUTS, pi, s5)
    h_s_56_prof[i] = AS_56_isentrop.hmass()
    T_s_56_prof[i] = AS_56_isentrop.T()
    rho_s_56_prof[i] = AS_56_isentrop.rhomass()

#Zustandsgrößen setzen
AS_6_isentrop = AbstractState('HEOS', fluid)
T6s = T_s_56_prof[-1]
h6_s = h_s_56_prof[-1]
p6_s = p_s_56_prof[-1]
s6 = s5
AS_6_isentrop.update(CP.PSmass_INPUTS, p6_s, s6)
rho_6= AS_6_isentrop.rhomass()


# 4-->5 isobare Wärmezufuhr
Q_dot_zu = m_flow * (h5-h4)

#6-->7 Zustand 7s definieren
AS_7_isentrop = AbstractState('HEOS', fluid)
AS_7_real = AbstractState('HEOS', fluid)
AS_7_isentrop.update(CP.PSmass_INPUTS, p7_s, s6)
h7_s = AS_7_isentrop.hmass()

# Reales h ausrechnen und realen Zustand definieren
h7 = h6_s + (h7_s - h6_s)/eta_Luefter
AS_7_real.update(CP.HmassP_INPUTS, h7, p7_s)
T7 = AS_7_real.T()
s7 = AS_7_real.smass()

# 6-->7: Profil über Druck (isentrop + real)

# Druckprofil von 6 nach 7 (einfach linear aufgelöst)
N_67 = 200
p_67_prof = np.linspace(p6_s, p7_s, N_67)

AS_67_isentrop = AbstractState('HEOS', fluid)
AS_67_real = AbstractState('HEOS', fluid)

# Arrays anlegen
#Isentrop
T_s_67_prof = np.empty_like(p_67_prof)
h_s_67_prof = np.empty_like(p_67_prof)
#Realer Fall
T_67_prof = np.empty_like(p_67_prof)
h_67_prof = np.empty_like(p_67_prof)
s_67_prof = np.empty_like(p_67_prof)
rho_67_prof = np.empty_like(p_67_prof)

for i, pi in enumerate(p_67_prof):
    # --- isentroper Referenzpfad (s = s6 konstant) ---
    AS_67_isentrop.update(CP.PSmass_INPUTS, pi, s6)
    h_s_loc = AS_67_isentrop.hmass()
    T_s_67_prof[i] = AS_67_isentrop.T()
    h_s_67_prof[i] = h_s_loc

    # --- realer Pfad mit Lüfterwirkungsgrad ---
    # Bezug immer auf den Inlet-Zustand 6s
    h_real_loc = h6_s + (h_s_loc - h6_s)/eta_Luefter
    AS_67_real.update(CP.HmassP_INPUTS, h_real_loc, pi)
    T_67_prof[i] = AS_67_real.T()
    h_67_prof[i] = h_real_loc
    s_67_prof[i] = AS_67_real.smass()
    rho_67_prof[i] = AS_67_real.rhomass()

# Endzustände aus Profil übernehmen
p7   = p_67_prof[-1]
T7   = T_67_prof[-1]
h7   = h_67_prof[-1]
s7   = s_67_prof[-1]
rho_7 = rho_67_prof[-1]

# Isentroper Endzustand 7s
T7_s = T_s_67_prof[-1]
h7_s = h_s_67_prof[-1]

# Tabelle als Liste von Dictionaries
zustands_tabelle = [
    {"Zustand": "1",  "p [Pa]": p1,     "T [K]": T1,     "h [J/kg]": h1,     "s [J/kgK]": s1},
    {"Zustand": "2s", "p [Pa]": p2_s,   "T [K]": T2_s,   "h [J/kg]": h2_s,   "s [J/kgK]": s1},
    {"Zustand": "2",  "p [Pa]": p2,     "T [K]": T2,     "h [J/kg]": h2,     "s [J/kgK]": s2},
    {"Zustand": "3",  "p [Pa]": p3,     "T [K]": T3,     "h [J/kg]": h3,     "s [J/kgK]": s3},
    {"Zustand": "4s", "p [Pa]": p4_s,   "T [K]": T4_s,   "h [J/kg]": h4_s,   "s [J/kgK]": s3},
    {"Zustand": "4",  "p [Pa]": p4,     "T [K]": T4,     "h [J/kg]": h4,     "s [J/kgK]": s4},
    {"Zustand": "5",  "p [Pa]": p5,     "T [K]": T5,     "h [J/kg]": h5,     "s [J/kgK]": s5},
    {"Zustand": "6s", "p [Pa]": p6_s,   "T [K]": T6s,    "h [J/kg]": h6_s,   "s [J/kgK]": s5},
    {"Zustand": "7",  "p [Pa]": p7_s,   "T [K]": T7,     "h [J/kg]": h7,     "s [J/kgK]": s7},
]

print(f'{m_flow}')

df = pd.DataFrame(zustands_tabelle)

print(df)

#Leistugen berechnen
P_Luefter =  m_flow * (h7-h6_s)
P_Expansion34 = m_flow * (h4-h3)
P_Verdichtung = m_flow * (h2-h1)
P_Expansion56 = m_flow * (h6_s - h5)
P_Verdichtung71 = m_flow * (h1 - h7)

HS_1 = P_Verdichtung + P_Luefter + P_Expansion34 + Q_dot_zu + Q_dot_ab + P_Verdichtung71 + P_Expansion56
COP = -Q_dot_ab / P_Luefter

#Quellenseitig
AS_Quelle = AbstractState('HEOS','Water')
# p_Quelle usw. definieren

p_Quelle = 150000 #[Pa]
T_Quelle_ein = 283.15

#Zustand beim Eintritt setzen
AS_Quelle.update(CP.PT_INPUTS, p_Quelle, T_Quelle_ein)

# cp definieren
cp_Quelle = AS_Quelle.cpmass()

# Bennötigter Massestrom Quelle

m_flow_Quelle = Q_dot_zu/(cp_Quelle*3)
m_flow_Senke = -Q_dot_ab/(cp_Quelle*15)

#Volumenstrom vor Lüfter

V_dot = m_flow/rho_6

print(f'{m_flow}, V_dot = {V_dot} P_Verd = {P_Verdichtung}, Q_ab = {Q_dot_ab},P_Exp_34 = {P_Expansion34}')
print(f'Q_zu = {Q_dot_zu}, P_Exp_56 = {P_Expansion56}, P_Lüfter = {P_Luefter}, P_Verd_71 ={P_Verdichtung71}')
print(f'Erster Hauptsatz check = {HS_1}, {COP} , {m_flow_Quelle} , {m_flow_Senke})')

print('Dichten über die Zustandspunkte 1-7')
print(f'rho_1 = {rho_1} ; rho_2 = {rho_2} ; rho_3 = {rho_3}; rho_4 = {rho_4} ; rho_5 = {rho_5}; rho_6 = {rho_6}; rho_7 = {rho_7}')
#============= ab hier : Plotts =============================================================================


# ============================================================
# log(p)-h Diagramm des Kreisprozesses
# ============================================================


def plot_log_ph_diagramm():
    # --- Hilfsfunktionen für Einheiten ---
    def h_kJ(h):   # J/kg -> kJ/kg
        return h / 1e3

    def p_bar(p):  # Pa -> bar
        return p / 1e5

    fig, ax = plt.subplots(figsize=(8, 6))

    # ================================
    # 1 -> 2 (Verdichtung)
    # ================================
    # realer Pfad
    ax.semilogy(h_kJ(h_12_prof), p_bar(p_12_prof),
                label="1-2 real", linewidth=2)
    # isentroper Referenzpfad
    ax.semilogy(h_kJ(h_s_12_prof), p_bar(p_s_12_prof),
                "--", label="1-2 s")

    # ================================
    # 2 -> 3 (Wärmeabfuhr, isobar)
    # ================================
    ax.semilogy([h_kJ(h2), h_kJ(h3)],
                [p_bar(p2), p_bar(p3)],
                label="2-3 isobar", linewidth=2)

    # ================================
    # 3 -> 4 (Expansion)
    # ================================
    # real
    ax.semilogy(h_kJ(h_34_prof), p_bar(p_34_prof),
                label="3-4 real", linewidth=2)
    # isentrop
    ax.semilogy(h_kJ(h_s_34_prof), p_bar(p_s_34_prof),
                "--", label="3-4 s")

    # ================================
    # 4 -> 5 (Wärmezufuhr, isobar)
    # ================================
    ax.semilogy([h_kJ(h4), h_kJ(h5)],
                [p_bar(p4), p_bar(p5)],
                label="4-5 isobar", linewidth=2)

    # ================================
    # 5 -> 6 (rotierende isentrope Strecke)
    # ================================
    ax.semilogy(h_kJ(h_s_56_prof), p_bar(p_s_56_prof),
                label="5-6 s", linewidth=2)

    # ================================
    # 6 -> 7 (Lüfterstufe)
    # ================================
    try:
        # Wenn du das 6-7-Profil schon berechnet hast:
        ax.semilogy(h_kJ(h_67_prof), p_bar(p_67_prof),
                    label="6-7 real", linewidth=2)
        ax.semilogy(h_kJ(h_s_67_prof), p_bar(p_67_prof),
                    "--", label="6-7 s")
    except NameError:
        # Fallback: nur Verbindung zwischen Punkten
        ax.semilogy([h_kJ(h6_s), h_kJ(h7)],
                    [p_bar(p6_s), p_bar(p7)],
                    label="6-7 (ohne Profil)", linewidth=2)

    # ================================
    # 7 -> 1 (Wärmeaufnahme, "Rückführung")
    # ================================
    ax.semilogy([h_kJ(h7), h_kJ(h1)],
                [p_bar(p7), p_bar(p1)],
                label="7-1", linewidth=2)

    # ================================
    # Zustands­punkte markieren
    # ================================
    states_h = [h1, h2, h3, h4, h5, h6_s, h7]
    states_p = [p1, p2, p3, p4, p5, p6_s, p7]
    labels    = ["1", "2", "3", "4", "5", "6", "7"]

    for hi, pi, lab in zip(states_h, states_p, labels):
        ax.semilogy(h_kJ(hi), p_bar(pi), "ko")        # schwarzer Punkt
        ax.text(h_kJ(hi) * 1.001, p_bar(pi) * 1.01,
                lab, fontsize=10)

    # ================================
    # Achsen, Gitter, Layout
    # ================================
    ax.set_xlabel("Enthalpie h [kJ/kg]")
    ax.set_ylabel("Druck p [bar]")
    ax.set_title("log(p)-h Diagramm der Rotationswärmepumpe")

    ax.grid(True, which="both", linestyle=":", linewidth=0.7)
    ax.legend(loc="best")
    fig.tight_layout()

    return fig, ax

# Aufruf
fig_ph, ax_ph = plot_log_ph_diagramm()
plt.show()


def plot_Ts_diagramm():
    # Entropie in kJ/(kg*K) umrechnen
    def s_kJ(s):
        return s / 1e3  # J/(kg*K) -> kJ/(kg*K)

    fig, ax = plt.subplots(figsize=(8, 6))

    # ================================
    # 1 -> 2 (reale Verdichtung)
    # ================================
    ax.plot(s_kJ(s_12_prof), T_12_prof, label="1-2 real", linewidth=2)

    # isentroper Referenzpfad 1-2s (s = s1 konstant)
    s_12_s_prof = np.full_like(T_s_12_prof, s1)
    ax.plot(s_kJ(s_12_s_prof), T_s_12_prof, "--", label="1-2 s")

    T2_s = T_s_12_prof[-1]
    s2_s = s1

    # ================================
    # 2 -> 3 (Wärmeabfuhr, isobar)
    # ================================
    ax.plot([s_kJ(s2), s_kJ(s3)],
            [T2, T3],
            label="2-3 (Wärmeabfuhr)", linewidth=2)

    # ================================
    # 3 -> 4 (reale Expansion)
    # ================================
    ax.plot(s_kJ(s_34_prof), T_34_prof, label="3-4 real", linewidth=2)

    # isentroper Referenzpfad 3-4s (s = s3 konstant)
    s_34_s_prof = np.full_like(T_s_34_prof, s3)
    ax.plot(s_kJ(s_34_s_prof), T_s_34_prof, "--", label="3-4 s")

    T4_s = T_s_34_prof[-1]
    s4_s = s3

    # ================================
    # 4 -> 5 (isobare Wärmezufuhr)
    # ================================
    ax.plot([s_kJ(s4), s_kJ(s5)],
            [T4, T5],
            label="4-5 (Wärmezufuhr)", linewidth=2)

    # ================================
    # 5 -> 6 (isentrop im Rotor)
    # ================================
    s_56_prof = np.full_like(T_s_56_prof, s5)
    ax.plot(s_kJ(s_56_prof), T_s_56_prof, label="5-6 s", linewidth=2)

    # ================================
    # 6 -> 7 (Lüfterstufe)
    # ================================
    try:
        ax.plot(s_kJ(s_67_prof), T_67_prof, label="6-7 real", linewidth=2)
    except NameError:
        ax.plot([s_kJ(s6), s_kJ(s7)],
                [T6s, T7],
                label="6-7", linewidth=2)

    # ================================
    # 7 -> 1 (Wärmeaufnahme / Rückführung)
    # ================================
    ax.plot([s_kJ(s7), s_kJ(s1)],
            [T7, T1],
            label="7-1 (Wärmeaufnahme)", linewidth=2)

    # ================================
    # Wasser-Seite: Senke & Quelle
    # ================================
    T3_w = 273.15 + 35.0    # 308.15 K
    T2_w = 323.15           # 50 °C
    T4_w = 280.15
    T5_w = 283.15

    # Senke (oben, 3 -> 2)
    ax.plot([s_kJ(s3), s_kJ(s2)],
            [T3_w, T2_w],
            color="tab:orange",
            linewidth=2,
            label="Senke (Wasser)")

    # Quelle (unten, 4 -> 5)
    ax.plot([s_kJ(s4), s_kJ(s5)],
            [T4_w, T5_w],
            color="tab:blue",
            linewidth=2,
            label="Quelle (Wasser)")

    # ================================
    # Zustands­punkte markieren
    # ================================
    # ================================
    # Zustands­punkte markieren
    # ================================
    # reale Zustände 1..7
    states_T_real = [T1, T2, T3, T4, T5, T6s, T7]
    states_s_real = [s1, s2, s3, s4, s5, s6, s7]
    labels_real = ["1", "2", "3", "4", "5", "6", "7"]

    for Ti, si, lab in zip(states_T_real, states_s_real, labels_real):
        x = s_kJ(si)
        y = Ti
        ax.plot(x, y, "ko")
        ax.annotate(
            lab,
            (x, y),
            xytext=(4, 4),  # Offset in Punkten (rechts/oben)
            textcoords="offset points",
            fontsize=10
        )



    # ================================
    # Achsen, Gitter, Legende außen
    # ================================
    ax.set_xlabel("Entropie s [kJ/(kg·K)]")
    ax.set_ylabel("Temperatur T [K]")
    ax.set_title("T-s Diagramm der Rotationswärmepumpe")

    # Temperaturbereich automatisch aus Daten holen
    all_T = np.array([T1, T2, T3, T4, T5, T6s, T7,
                      283.15, 281.15])  # + Wasserlinien, falls du willst
    T_min = np.floor(all_T.min() / 5) * 5 - 5
    T_max = np.ceil(all_T.max()  / 5) * 5 + 5

    ax.set_ylim(T_min, T_max)

    # Major-Ticks alle 5 K, Minor-Ticks alle 1 K
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(1))

    # Grid für Major und Minor
    ax.grid(which="major", linestyle=":", linewidth=0.7)
    ax.grid(which="minor", linestyle=":", linewidth=0.3, alpha=0.4)

    # Legende rechts außen
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.,
        fontsize=9
    )

    cop_text = rf"COP = {COP:.2f}"
    ax.text(
        1.02, 0.55,  # x, y in Achsenkoordinaten (rechts, etwas unterhalb der Legende)
        cop_text,
        transform=ax.transAxes,
        fontsize=13,
        va="top"
    )
    # ṁ-Text noch etwas darunter
    mdot_text = rf"$\dot{{m}}_\mathrm{{{fluid}}} = {m_flow:.3f}\,\mathrm{{kg/s}}$"
    ax.text(
        1.02, 0.5,
        mdot_text,
        transform=ax.transAxes,
        fontsize=13,
        va="top"
    )

    # ṁ-Senke-Text noch etwas darunter
    mdot_text = rf"$\dot{{m}}_\mathrm{{Senke}} = {m_flow_Senke:.3f}\,\mathrm{{kg/s}}$"
    ax.text(
        1.02, 0.45,
        mdot_text,
        transform=ax.transAxes,
        fontsize=13,
        va="top"
    )

    # ṁ-Quelle-Text noch etwas darunter
    mdot_text = rf"$\dot{{m}}_\mathrm{{Quelle}} = {m_flow_Quelle:.3f}\,\mathrm{{kg/s}}$"
    ax.text(
        1.02, 0.4,
        mdot_text,
        transform=ax.transAxes,
        fontsize=13,
        va="top"
    )

    # Platz für die Legende rechts lassen
    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])

    return fig, ax

# Aufruf:
fig_ts, ax_ts = plot_Ts_diagramm()
plt.show()