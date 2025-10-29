# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, AutoMinorLocator
from CoolProp import CoolProp as CP
from CoolProp.CoolProp import AbstractState

# -----------------------------
# Parameter
# -----------------------------
fluid = 'Argon'
xmax_kJ = 100.0
n_iso_T = 12
n_iso_s = 10

AS = AbstractState('HEOS', fluid)

# Referenzzustand
T_ref, p_ref = 298.15, 1e5
h_ref = CP.PropsSI('H', 'T', T_ref, 'P', p_ref, fluid)

# Grenzen
T_triple = CP.PropsSI('Ttriple', fluid)
T_crit   = CP.PropsSI('Tcrit', fluid)
p_triple = CP.PropsSI('ptriple', fluid)
p_crit   = CP.PropsSI('pcrit', fluid)

T_min = T_triple * 1.05
T_max = 373.15  # bis 100 °C
p_min = p_triple
p_max = p_crit * 2.0

# -----------------------------
# Sättigungslinien (bis knapp < T_crit)
# -----------------------------
T_sat = np.linspace(T_min, T_crit*0.999, 500)
p_sat = np.array([CP.PropsSI('P', 'T', T, 'Q', 0, fluid) for T in T_sat])
h_f   = np.array([CP.PropsSI('H', 'T', T, 'Q', 0, fluid) - h_ref for T in T_sat])
h_g   = np.array([CP.PropsSI('H', 'T', T, 'Q', 1, fluid) - h_ref for T in T_sat])

# Kritischer Punkt anfügen -> Glocke schließen
AS.update(CP.PT_INPUTS, p_crit, T_crit)
h_crit = AS.hmass() - h_ref
p_sat  = np.append(p_sat, p_crit)
h_f    = np.append(h_f, h_crit)
h_g    = np.append(h_g, h_crit)

# -----------------------------
# Raster für Isothermen / Isentropen
# -----------------------------
T_iso  = np.linspace(T_min, T_max, n_iso_T)
p_vals = np.logspace(np.log10(p_min*1.1), np.log10(p_max), 450)

AS.update(CP.PT_INPUTS, p_max*0.8, max(T_min*1.2, T_triple*1.1))
s_min = AS.smass()
AS.update(CP.PT_INPUTS, max(p_min*5, p_ref), min(T_ref*1.4, T_max))
s_max = AS.smass()
s_iso = np.linspace(s_min, s_max, n_iso_s)

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(14.5, 7.5))

# Nassdampfgebiet
ax.fill_betweenx(p_sat/1e5, h_f/1e3, h_g/1e3,
                 color='lightblue', alpha=0.3, label='Nassdampfgebiet')

# Sättigungslinien schwarz + kritischer Punkt
ax.semilogy(h_f/1e3, p_sat/1e5, 'k-', lw=1.6, label='gesättigte Flüssigkeit')
ax.semilogy(h_g/1e3, p_sat/1e5, 'k-', lw=1.6, label='gesättigter Dampf')
ax.semilogy([h_crit/1e3], [p_crit/1e5], 'ko', ms=5)
ax.text(h_crit/1e3, p_crit/1e5, '  kritisch', va='center', ha='left', fontsize=8)

# Hilfsfunktion für Labels (mit optionalem Offset)
def label_line_endpoints(ax, x_arr, y_arr, text, color, dx=0, dy=0):
    x_arr = np.asarray(x_arr); y_arr = np.asarray(y_arr)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mask.sum() < 2: return
    xv, yv = x_arr[mask], y_arr[mask]
    ax.text(xv[5]+dx,  yv[5]* (1+dy),  text, fontsize=7, color=color,
            ha='left', va='bottom', clip_on=True)
    ax.text(xv[-5]+dx, yv[-5]* (1-dy), text, fontsize=7, color=color,
            ha='right', va='top',    clip_on=True)

# -----------------------------
# Isothermen (rot)
# -----------------------------
for i, T in enumerate(T_iso):
    h_iso = []
    for p in p_vals:
        try:
            AS.update(CP.PT_INPUTS, p, T)
            h_iso.append(AS.hmass() - h_ref)
        except ValueError:
            h_iso.append(np.nan)
    h_iso = np.asarray(h_iso)
    ax.semilogy(h_iso/1e3, p_vals/1e5, '-', lw=0.9, color='red', alpha=0.95,
                label='Isothermen' if i == 0 else "")
    label_line_endpoints(ax, h_iso/1e3, p_vals/1e5, f'{T-273.15:.0f}°C', 'red', dx=0, dy=0)

# -----------------------------
# Isentropen (grün, gestrichelt, leicht versetzt)
# -----------------------------
for j, s in enumerate(s_iso):
    h_iso = []
    for p in p_vals:
        try:
            AS.update(CP.PSmass_INPUTS, p, s)
            h_iso.append(AS.hmass() - h_ref)
        except ValueError:
            h_iso.append(np.nan)
    h_iso = np.asarray(h_iso)
    ax.semilogy(h_iso/1e3, p_vals/1e5, '--', lw=0.85, color='green', alpha=0.9,
                label='Isentropen' if j == 0 else "")
    # Beschriftung leicht versetzt (nach oben + rechts)
    label_line_endpoints(ax, h_iso/1e3, p_vals/1e5,
                         f'{s/1e3:.2f} kJ/kgK', 'green', dx=1.5, dy=0.07)

# -----------------------------
# Achsen, Ticks, Grid
# -----------------------------
xmin_kJ = (min(np.nanmin(h_f), np.nanmin(h_g))/1e3) - 0.2*abs(np.nanmin(h_f)/1e3)
ax.set_xlim(xmin_kJ, xmax_kJ)
ax.set_ylim(p_min/1e5, p_max/1e5)

# Logarithmische y-Ticks
ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(1,10)*0.1))
ax.yaxis.set_minor_formatter(plt.NullFormatter())

# Minor-Ticks für x-Achse
ax.xaxis.set_minor_locator(AutoMinorLocator(5))

# Labels & Layout
ax.set_xlabel('spezifische Enthalpie h - h_ref [kJ/kg]')
ax.set_ylabel('Druck p [bar]')
ax.set_title(f'log(p)-h-Diagramm für {fluid} bis 100°C (h_ref bei 25°C, 1 bar = 0)')

ax.grid(True, which='both', linestyle=':', alpha=0.7)
ax.legend(loc='lower left', framealpha=0.9)
fig.tight_layout()
plt.show()
