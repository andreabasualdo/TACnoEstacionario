#%%
#Libraries
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({'font.size': 13, 'lines.linewidth': 2.5})
from matplotlib.widgets import Slider, Button, TextBox
import pandas as pd
import os
"""
COMPUESTOS QUE PARTICIPAN DE LAS REACCIONES: A,B,C,D,Z,G
PRODUCTO PRINCIPAL: C
REACTIVO PRINCIPAL: A
COMPUESTOS INERTES: M

ECUACIONES:
A + B ---> C + D
C --->  Z
A --->  G
"""

#%%
#Explicit equations
# Parámetros generales
UA = 10000  # [BTU/h·°F] coef de transferencia de calor
R = 1.987   # [BTU/lbmol·R] constante de los gases
V = 80      # [ft3] Volumen del reactor

# Parámetros de la reacción 1 (siempre activa)
A = 9000000
E = 17000  # [BTU/lbmol]
dH = -30000  # [BTU/lbmol]

# Parámetros de la reacción 2 (puede activarse o no)
reaccion_2 = True
if reaccion_2:
    A2 = 200000
    E2 = 15000
    dH2 = -10000
    c2, z2 = 1, 1  # Coeficientes estequiométricos
    ordenc2=1
else:
    A2 = dH2 = c2 = z2 = 0
    E2 = 100000000000
    ordenc2=0

# Parámetros de la reacción 3 (puede activarse o no)
reaccion_3 = False
if reaccion_3:
    A3 = 1
    E3 = 5
    dH3 = 1
    a3, g3 = 1, 1  # Coeficientes estequiométricos
    ordena3=0
else:
    A3 = dH3 = a3 = g3 = 0
    E3 = 1000000000000
    ordena3=0

# Condiciones iniciales
Ta1 = 40  # [°F]
CpW = 20  # [BTU/lbmol·°F]
mc = 3000  # [lb-mol/hora]
T0 = 77  # [°F]
Fa0 = 15  # [lb-mol/hora]
Fb0 = 25  # [lb-mol/hora]

#Flujo de entrada al reactor
T0=80             #[°F]temperatura de la corriente que ingresa al reactor
Fa0=35       #[lb-mol/hora] Flujo de A en la corriente de ingreso
Fb0=105  #[lb-mol/hora] Flujo de B en la corriente de ingreso
Fm0=0           #[lb-mol/hora] Flujo del inerte en la corriente de ingreso
Fc0=0             #[lb-mol/hora] Flujo de C en la corriente de ingreso
roa=1.2         #[lb-mol/ft3] densidad de A
rob=4          #[lb-mol/ft3] densidad de B
rom=1          #[lb-mol/ft3] densidad del inerte M

#Condiciones iniciales en el reactor
Cai=0           #[lb-mol/ft3]concentracion de A en el reactor en tiempo t=0
Cbi=2          #[lb-mol/ft3]concentracion de B en el reactor en tiempo t=0
Cmi=0             #[lb-mol/ft3]concentracion de M en el reactor en tiempo t=0
Ti=100             #[°F]temperatura inicial del reactor
Cci=0             #[lb-mol/ft3]concentracion de C en el reactor en tiempo t=0  
Cdi=0             #[lb-mol/ft3]concentracion de D en el reactor en tiempo t=0
Czi=0             #[lb-mol/ft3]concentracion de Z en el reactor en tiempo t=0
Cgi=0             #[lb-mol/ft3]concentracion de G en el reactor en tiempo t=0

#orden de la reaccion
ordena=1
ordenb=0

#coeficientes estequiometricos
a=1
b=1
c=1
d=1
c2=1
z2=1
a3=0
g3=0

#Cp de los compuestos
CPa=28            #[BTU/lbmol·°F] cp de A
CPb=22            #[BTU/lbmol·°F] cp de B
CPc=38            #[BTU/lbmol·°F] cp de C
CPm=30           #[BTU/lbmol·°F] cp de M
CPd=16            #[BTU/lbmol·°F] cp de D
Cpz=20            #[BTU/lbmol·°F] cp de Z
Cpg=10



def ODEfun(Yfuncvec,t,UA,Ta1,CpW,T0,mc,dH,dH2,dH3,V,Fa0,Fb0,Fm0,Fc0,Cai,Cmi,roa,rob,rom,ordena,ordenb,
           ordenc2,ordena3,a,b,c,d,c2,z2,a3,g3,A,E,A2,E2,A3,E3,R,CPa,CPb,CPc,CPd,CPm,Cpz,Cpg):

    Ca= Yfuncvec[0]
    Cb= Yfuncvec[1]
    Cc= Yfuncvec[2]
    Cm= Yfuncvec[3]
    T= Yfuncvec[4]
    Cd= Yfuncvec[5]
    Cz= Yfuncvec[6]
    Cg= Yfuncvec[7]
    
    # Cálculo de la velocidad de la reacción 1
    k = A * np.exp(-E / R / (T + 460))
    Cam = pow(Ca, ordena)
    Cbn = pow(Cb, ordenb)
    r = k * Cam * Cbn  # Velocidad de reacción 1

    # Cálculo de la reacción 2 (si c2 y z2 son distintos de 0)
    if c2 != 0 and z2 != 0:
        k2 = A2 * np.exp(-E2 / R / (T + 460))
        Ccm = max(Cc,0)**2
        r2 = k2 * Ccm
        rc2 = -(1 / c2) * r2  # Tasa de consumo de C en la reacción 2
        rz = (1 / z2) * r2  # Tasa de generación de Z en la reacción 2
    else:
        r2 = 0
        rc2 = 0
        rz = 0

    # Cálculo de la reacción 3 (si a3 y g3 son distintos de 0)
    if a3 != 0 and g3 != 0:
        k3 = A3 * np.exp(-E3 / R / (T + 460))
        Cam2 = pow(Ca, ordena3)
        r3 = k3 * Cam2
        ra3 = -(1 / a3) * r3  # Tasa de consumo de A en la reacción 3
        rg = (1 / g3) * r3  # Tasa de generación de G en la reacción 3
    else:
        r3 = 0
        ra3 = 0
        rg = 0

    # Tasa de consumo/generación total de cada especie
    ra1=-(1 / a) * r
    ra = ra1 + ra3 if a != 0 else ra3
    rb = -(1 / b) * r if b != 0 else 0
    rc1 = (1 / c) * r if c != 0 else 0
    rc = rc1 + rc2  # Consumo total de C
    rd = (1 / d) * r if d != 0 else 0
    

    # Cálculo de los moles en el reactor
    Na, Nb, Nc, Nm, Nd, Nz, Ng = Ca * V, Cb * V, Cc * V, Cm * V, Cd * V, Cz * V, Cg * V

    # Cálculo de los flujos de entrada
    v0 = Fa0 / roa + Fb0 / rob + Fm0 / rom
    Ca0, Cb0, Cm0, Cc0 = Fa0 / v0, Fb0 / v0, Fm0 / v0, Fc0 / v0
    tau = V / v0

    # Cálculo de NCp (térmica total del reactor)
    NCp = Na * CPa + Nb * CPb + Nc * CPc + Nm * CPm + Nd * CPd + Nz * Cpz + Ng * Cpg

    # Balance de energía
    ThetaCp = CPa + (Fb0 / Fa0) * CPb + (Fm0 / Fa0) * CPm
    Ta2 = T - ((T - Ta1) * np.exp(-UA / (CpW * mc)))
    Qr2 = mc * CpW * (Ta2 - Ta1)
    Qr1 = Fa0 * ThetaCp * (T - T0)
    Qr = Qr1 + Qr2
    Qg = (ra1 * V * dH) + (rc2 * V * dH2) + (ra3 * V * dH3)

    # Ecuaciones diferenciales
    dCadt = ((Ca0 - Ca) / tau) + ra
    dCbdt = ((Cb0 - Cb) / tau) + rb
    dCcdt = ((Cc0 - Cc) / tau) + rc
    dCmdt = ((Cm0 - Cm) / tau)
    dCddt = ((0 - Cd) / tau) + rd
    dCzdt = ((0 - Cz) / tau) + rz
    dCgdt = ((0 - Cg) / tau) + rg
    dTdt = (Qg - Qr) / NCp  # Balance de energía

    return np.array([dCadt, dCbdt, dCcdt, dCmdt, dTdt, dCddt, dCzdt, dCgdt])

tspan = np.linspace(0, 6, 500) # Rangos para la variable independiente
y0 = np.array([Cai,Cbi,Cci,Cmi,Ti,Cdi,Czi,Cgi]) # Arreglo de valores iniciales de las variables dependientes en el reactor

#%%
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
fig.suptitle("""Simulacion TAC reacciones multiples (Unidades Inglesas)""", fontweight='bold', x = 0.26, y=1)
fig.subplots_adjust(wspace=0.35,hspace=0.3, top=0.33)

# Resolver la ODE
sol = odeint(ODEfun, y0, tspan, (UA, Ta1, CpW, T0, mc, dH, dH2, dH3, V, Fa0, Fb0, Fm0, Fc0, Cai, Cmi, roa, rob, rom, 
                                 ordena, ordenb, ordenc2, ordena3, a, b, c, d, c2, z2, a3, g3, A, E, A2, E2, A3, E3, R, 
                                 CPa, CPb, CPc, CPd, CPm, Cpz, Cpg))

Ca = sol[:, 0]
Cb= sol[:, 1]
Cc= sol[:, 2]
Cm= sol[:, 3]
T=sol[:, 4]
Cd=sol[:, 5]
Cz=sol[:, 6]
Cg=sol[:, 7]

# Cálculo del caudal volumétrico de entrada
v0 = Fa0 / roa + Fb0 / rob + Fm0 / rom
Ca0, Cb0, Cm0, Cc0 = Fa0 / v0, Fb0 / v0, Fm0 / v0, Fc0 / v0
tau = V / v0

# Conversión de A
X = (Ca0 - Ca) / Ca0  

# Selectividad 
if c2 != 0 or a3 != 0:  # Solo calcular S si hay al menos una reacción adicional
    S = ((Cc0 - Cc) * (-a)) / ((Ca0 - Ca) * c)
else:
    S = np.full_like(tspan, 0)  

# Constantes de velocidad y velocidades de reacción
k = A * np.exp(-E / R / (T + 460))
r = k * np.power(Ca, ordena) * np.power(Cb, ordenb)  

ra1 = -(1 / a) * r if a != 0 else 0
rc1 = (1 / c) * r if c != 0 else 0
rb = -(1 / b) * r if b != 0 else 0
rd = (1 / d) * r if d != 0 else 0

# Solo calcular reacciones 2 y 3 si están activas
if c2 != 0:
    k2 = A2 * np.exp(-E2 / R / (T + 460))
    r2 = k2 * np.power(Cc, ordenc2)
    rc2 = -(1 / c2) * r2
    rz = (1 / z2) * r2 if z2 != 0 else 0
else:
    rc2 = rz = 0

if a3 != 0:
    k3 = A3 * np.exp(-E3 / R / (T + 460))
    r3 = k3 * np.power(Ca, ordena3)
    ra3 = -(1 / a3) * r3
    rg = (1 / g3) * r3 if g3 != 0 else 0
else:
    ra3 = rg = 0

# Sumar tasas de reacción totales
ra = ra1 + ra3
rc = rc1 + rc2

# Balance de energía
ThetaCp = CPa + (Fb0 / Fa0) * CPb + (Fm0 / Fa0) * CPm
Ta2 = T - ((T - Ta1) * np.exp(-UA / (CpW * mc)))
Qr2 = mc * CpW * (Ta2 - Ta1)
Qr1 = Fa0 * ThetaCp * (T - T0)
Qr = Qr1 + Qr2
Qg = (ra1 * V * dH) + (rc2 * V * dH2) + (ra3 * V * dH3)  


p1, p1c, p1z, p1g= ax1.plot(tspan, Ca, tspan, Cc, tspan, Cz, tspan, Cg)
ax1.legend([r'$C_A$', r'$C_C$'], loc='upper left', fontsize='xx-small')  # Actualiza la leyenda
ax1.set_xlabel('time $(hr)$', fontsize='medium')
ax1.set_ylabel(r'Concentraciones(lb-mol/$ft^{3}$)', fontsize='medium')
ax1.grid()
ax1.set_ylim(0, 0.5)
ax1.set_xlim(0, 5)

p2 = ax2.plot(tspan, T)[0]
ax2.legend([r'$T$'], loc='upper right')
ax2.set_xlabel('time $(hr)$', fontsize='medium')
ax2.set_ylabel(r'Temperature $(^\circ F)$', fontsize='medium')
ax2.grid()
ax2.set_ylim(60, 250)
ax2.set_xlim(0, 5)

p3, p3s = ax3.plot(tspan,X, tspan, S)
ax3.set_xlabel(r'Time $(hr)$', fontsize='medium')
ax3.set_ylabel(r'$X_A$ & $S_C$', fontsize='medium')
ax3.grid()
ax3.set_ylim(0, 1)
ax3.set_xlim(0, 5)

p4,p5 = ax4.plot(tspan,Qg,tspan,Qr)
ax4.set_xlabel('time $(hr)$', fontsize='medium')
ax4.legend(['$Q_g$','$Q_r$'], loc='upper right')
ax4.set_ylabel(r'$Q$(Btu/$hr$)', fontsize='medium')
ax4.grid()
ax4.ticklabel_format(style='sci',scilimits=(3,4),axis='y')
ax4.set_ylim(0, 8*10**6)
ax4.set_xlim(0, 5)

"""1.text(-2.5,0.69,'Reacciones:'
     '\n\n'
         r'1)  aA + bB --> cC + dD'
          , ha='left', wrap = True, fontsize=13, fontweight='bold')
ax1.text(4.5,0.7,'Orden de la rx' '\n'
         r'r = k . Ca$^{m}$ . Cb$^{n}$', ha='left', wrap = True, fontsize=13, fontweight='bold')

if reaccion_2:
    ax1.text(-2.5,0.52,'' '\n'
         r' 2)  cC --> zZ', ha='left', wrap = True, fontsize=13, fontweight='bold')

if reaccion_3:        
    ax1.text(3.5,0.52,'' '\n'
         r' 3)  aA --> gG', ha='left', wrap = True, fontsize=13, fontweight='bold')
    
ax1.text(10,0.7,'Reactor:' '\n'
         r'', ha='left', wrap = True, fontsize=13, fontweight='bold')
ax1.text(10,0.45,'Intercambiador de calor:' '\n'
         r'', ha='left', wrap = True, fontsize=13, fontweight='bold')
ax1.text(16.5,0.7,'Corrientes de entrada:' '\n'
         r'', ha='left', wrap = True, fontsize=13, fontweight='bold')"""

axcolor = 'black'
ax_a = plt.axes([0.025, 0.83, 0.03, 0.015], facecolor=axcolor)
ax_b = plt.axes([0.085, 0.83, 0.03, 0.015], facecolor=axcolor)
ax_c = plt.axes([0.145, 0.83, 0.03, 0.015], facecolor=axcolor)
ax_d = plt.axes([0.205, 0.83, 0.03, 0.015], facecolor=axcolor)
ax_dH = plt.axes([0.09, 0.73, 0.1, 0.015], facecolor=axcolor)
ax_E = plt.axes([0.09, 0.79, 0.1, 0.015], facecolor=axcolor)
ax_A = plt.axes([0.09, 0.76, 0.1, 0.015], facecolor=axcolor)
ax_ordena = plt.axes([0.33, 0.84, 0.05, 0.015], facecolor=axcolor)
ax_ordenb = plt.axes([0.33, 0.81, 0.05, 0.015], facecolor=axcolor)
ax_V = plt.axes([0.55, 0.86, 0.1, 0.015], facecolor=axcolor)
ax_Cai = plt.axes([0.55, 0.82, 0.1, 0.015], facecolor=axcolor)
ax_Cbi = plt.axes([0.55, 0.78, 0.1, 0.015], facecolor=axcolor)
ax_Cmi = plt.axes([0.55, 0.74, 0.1, 0.015], facecolor=axcolor)
ax_Ti = plt.axes([0.55, 0.7, 0.1, 0.015], facecolor=axcolor)
ax_UA = plt.axes([0.55, 0.57, 0.1, 0.015], facecolor=axcolor)
ax_Ta1 = plt.axes([0.55, 0.53, 0.1, 0.015], facecolor=axcolor)
ax_CpW = plt.axes([0.55, 0.49, 0.1, 0.015], facecolor=axcolor)
ax_T0 = plt.axes([0.55, 0.45, 0.1, 0.015], facecolor=axcolor)
ax_mc = plt.axes([0.55, 0.41, 0.1, 0.015], facecolor=axcolor)
ax_Fa0 = plt.axes([0.82, 0.86, 0.1, 0.015], facecolor=axcolor)
ax_Fb0 = plt.axes([0.82, 0.82, 0.1, 0.015], facecolor=axcolor)
ax_Fm0 = plt.axes([0.82, 0.78, 0.1, 0.015], facecolor=axcolor)
ax_roa = plt.axes([0.82, 0.74, 0.1, 0.015], facecolor=axcolor)
ax_rob = plt.axes([0.82, 0.7, 0.1, 0.015], facecolor=axcolor)
ax_rom = plt.axes([0.82, 0.66, 0.1, 0.015], facecolor=axcolor)

print(ax_UA, ax_A, ax_E)

if reaccion_2:
    ax_c2 = plt.axes([0.045, 0.63, 0.03, 0.015], facecolor=axcolor)
    ax_z2 = plt.axes([0.115, 0.63, 0.03, 0.015], facecolor=axcolor)
    ax_E2 = plt.axes([0.045, 0.60, 0.1, 0.015], facecolor=axcolor)
    ax_A2 = plt.axes([0.045, 0.57, 0.1, 0.015], facecolor=axcolor)
    ax_dH2 = plt.axes([0.045, 0.54, 0.1, 0.015], facecolor=axcolor)
    ax_ordenc2 = plt.axes([0.115, 0.49, 0.05, 0.015], facecolor=axcolor)

if reaccion_3:
    ax_ordena3 = plt.axes([0.33, 0.49, 0.05, 0.015], facecolor=axcolor)
    ax_a3 = plt.axes([0.28, 0.63, 0.03, 0.015], facecolor=axcolor)
    ax_g3 = plt.axes([0.34, 0.63, 0.03, 0.015], facecolor=axcolor)
    ax_E3 = plt.axes([0.28, 0.60, 0.1, 0.015], facecolor=axcolor)
    ax_A3 = plt.axes([0.28, 0.57, 0.1, 0.015], facecolor=axcolor)
    ax_dH3 = plt.axes([0.28, 0.54, 0.1, 0.015], facecolor=axcolor)



# Sliders generales
sUA = Slider(ax_UA, r'$UA$($\frac{Btu}{h.^\circ F}$)', 1000, 30000, valinit=UA, valfmt='%1.0f')
sA = Slider(ax_A, r'$A$', A/100, A*100, valinit=A, valfmt='%1.0e')
sE = Slider(ax_E, r'$E$', 10000, E*1000, valinit=E, valfmt='%1.0e')
sTa1 = Slider(ax_Ta1, r'$T_{a1}$ ($^\circ F$)', 30, 100, valinit=Ta1, valfmt='%1.1f')
sCpW = Slider(ax_CpW, r'$C_{P_W}$($\frac{Btu}{lbmol.^\circ F}$)', 5, 60, valinit=CpW, valfmt='%1.0f')
sT0 = Slider(ax_T0, r'$T_{0}$($^\circ F$)', 50, 180, valinit=T0, valfmt='%1.0f')
smc = Slider(ax_mc, r'$m_{C}$($\frac{lbmol}{hr}$)', 100, 10000, valinit=mc, valfmt='%1.0f')
sdH = Slider(ax_dH, r'$\Delta H_{Rx}$ ($\frac{Btu}{lbmol\thinspace A}$)', -70000, -10000, valinit=dH, valfmt='%1.0f')
sV = Slider(ax_V, r'$V$ ($ft^3$)', 20, 100, valinit=V, valfmt='%1.1f')
sTi = Slider(ax_Ti, r'$T_i$ ($^\circ F$)', 20, 250, valinit=Ti, valfmt='%1.1f')
sFa0 = Slider(ax_Fa0, r'$F_{A0}$ ($\frac{lbmol}{hr}$)', 30, 150, valinit=Fa0, valfmt='%1.1f')
sFb0 = Slider(ax_Fb0, r'$F_{B0}$ ($\frac{lbmol}{hr}$)', 40, 2000, valinit=Fb0, valfmt='%1.0f')
sFm0 = Slider(ax_Fm0, r'$F_{M0}$ ($\frac{lbmol}{hr}$)', 0, 250, valinit=Fm0, valfmt='%1.1f')
sCai= Slider(ax_Cai,r'$C_{A_i}$ ($\frac{lbmol}{ft^{3}}$)', 0, 0.25, valinit= Cai,valfmt='%1.2f')
sCbi= Slider(ax_Cbi,r'$C_{B_i}$ ($\frac{lbmol}{ft^{3}}$)', 0, 0.25, valinit= Cbi,valfmt='%1.2f')
sCmi= Slider(ax_Cmi,r'$C_{M_i}$ ($\frac{lbmol}{ft^{3}}$)', 0, 0.25, valinit= Cmi,valfmt='%1.2f')
sroa= Slider(ax_roa,r'$ρ_{A}$ ($\frac{lbmol}{ft^{3}}$)', 0, 10, valinit= roa,valfmt='%1.2f')
srob= Slider(ax_rob,r'$ρ_{B}$ ($\frac{lbmol}{ft^{3}}$)', 0, 10, valinit= rob,valfmt='%1.2f')
srom= Slider(ax_rom,r'$ρ_{M}$ ($\frac{lbmol}{ft^{3}}$)', 0, 10, valinit= rom,valfmt='%1.2f')
# Sliders de órdenes de reacción (dependiendo de qué reacciones están activas)
sordena = Slider(ax_ordena, r'$m$', 0, 3, valinit=ordena, valfmt='%1.0f')
sordenb = Slider(ax_ordenb, r'$n$', 0, 3, valinit=ordenb, valfmt='%1.0f')

# Sliders adicionales para coeficientes (según corresponda)
sa = Slider(ax_a, r'$a$', 0, 3, valinit=a, valfmt='%1.0f', valstep=1)
sb = Slider(ax_b, r'$b$', 0, 3, valinit=b, valfmt='%1.0f', valstep=1)
sc = Slider(ax_c, r'$c$', 0, 3, valinit=c, valfmt='%1.0f', valstep=1)
sd = Slider(ax_d, r'$d$', 0, 3, valinit=d, valfmt='%1.0f', valstep=1)

# Sliders relacionados con reacciones secundarias (solo si ocurren)
if reaccion_2:
    sA2 = Slider(ax_A2, r'$A_{2}$', 1000, A2*100, valinit=A2, valfmt='%1.0e')
    sE2 = Slider(ax_E2, r'$E_{2}$', 10000, E2*1000, valinit=E2, valfmt='%1.0e')
    sdH2 = Slider(ax_dH2, r'$\Delta H_{Rx2}$', -10000, 50000, valinit=dH2, valfmt='%1.0f')
    sordenc2 = Slider(ax_ordenc2, r'$\mathbf{r_{2}=k_{2}.Cc^m}$   m', 0, 3, valinit=ordenc2, valfmt='%1.0f')
    sc2 = Slider(ax_c2, r'$c$', 0, 3, valinit=c2, valfmt='%1.0f', valstep=1)
    sz2 = Slider(ax_z2, r'$z$', 0, 3, valinit=z2, valfmt='%1.0f', valstep=1)

if reaccion_3:
    sA3 = Slider(ax_A3, r'$A_{3}$', 1000, A3*100, valinit=A3, valfmt='%1.0e')
    sE3 = Slider(ax_E3, r'$E_{3}$', 10000, 60000, valinit=E3, valfmt='%1.0e')
    sdH3 = Slider(ax_dH3, r'$\Delta H_{Rx3}$', -10000, 50000, valinit=dH3, valfmt='%1.0f')
    sordena3 = Slider(ax_ordena3, r'$\mathbf{r_{3}=k_{3}.Cc^n}$   n', 0, 3, valinit=ordena3, valfmt='%1.0f')
    sa3 = Slider(ax_a3, r'$a$', 0, 3, valinit=a3, valfmt='%1.0f', valstep=1)
    sg3 = Slider(ax_g3, r'$g$', 0, 3, valinit=g3, valfmt='%1.0f', valstep=1)
    


def update_plot2(val): #Actualizar las graficas segun valores ingresados

    print("¡update_plot2() se ejecutó!")
    print(f"UA: {sUA.val}, A: {sA.val}, E: {sE.val}, Ta1: {sTa1.val}, CpW: {sCpW.val}, T0: {sT0.val}, mc: {smc.val}")
    UA = sUA.val
    A = sA.val
    E = sE.val
    Ta1 =sTa1.val
    CpW = sCpW.val
    T0 =sT0.val
    mc = smc.val
    dH =sdH.val
    V = sV.val
    Ti=sTi.val
    Fa0 =sFa0.val
    Fb0 =sFb0.val
    Fm0 =sFm0.val
    Cai =sCai.val
    Cbi =sCbi.val
    Cmi =sCmi.val
    roa =sroa.val
    rob =srob.val
    rom =srom.val
    ordena=sordena.val
    ordenb=sordenb.val
    a=sa.val
    b=sb.val
    c=sc.val
    d=sd.val


    if reaccion_2:
        A2 = sA2.val
        E2 = sE2.val
        dH2 =sdH2.val
        c2=sc2.val
        z2=sz2.val
        ordenc2=sordenc2.val
    else:
        dH2 = 0
        A2 = 0
        E2 = 1e10
        c2 = 0
        z2 = 0
        ordenc2 = 0 

    if reaccion_3:
        A3 = sA3.val
        E3 = sE3.val
        dH3 =sdH3.val
        a3=sa3.val
        g3=sg3.val
        ordena3=sordena3.val
    else:
        dH3 = 0
        A3 = 0
        E3 = 1e10
        a3 = 0
        g3 = 0
        ordena3 = 0



    y1 = np.array([Cai,Cbi,Cci,Cmi,Ti,Cdi,Czi,Cgi]) 
    sol = odeint(ODEfun, y1, tspan, (UA,Ta1,CpW,T0,mc,dH,dH2,dH3,V,Fa0,Fb0,Fm0,Fc0,Cai,Cmi,roa,rob,rom,ordena,ordenb,ordenc2,ordena3,a,b,c,d,c2,z2,a3,g3,A,E,A2,E2,A3,E3,R,CPa,CPb,CPc,CPd,CPm,Cpz,Cpg))
    Ca = sol[:, 0]
    Cb= sol[:, 1]
    Cc= sol[:, 2]
    Cm= sol[:, 3]
    T=sol[:,4]
    Cd= sol[:, 5]
    Cz=sol[:, 6]
    Cg=sol[:, 7]

    # Cálculo del caudal volumétrico de entrada
    v0 = Fa0 / roa + Fb0 / rob + Fm0 / rom
    Ca0, Cb0, Cm0, Cc0 = Fa0 / v0, Fb0 / v0, Fm0 / v0, Fc0 / v0
    tau = V / v0

    # Conversión de A
    X = (Ca0 - Ca) / Ca0  

    # Selectividad 
    if c2 != 0 or a3 != 0:  # Solo calcular S si hay al menos una reacción adicional
        S = ((Cc0 - Cc) * (-a)) / ((Ca0 - Ca) * c)
    else:
        S = np.full_like(tspan, 0)  

    # Constantes de velocidad y velocidades de reacción
    k = A * np.exp(-E / R / (T + 460))
    r = k * np.power(Ca, ordena) * np.power(Cb, ordenb)  

    ra1 = -(1 / a) * r if a != 0 else 0
    rc1 = (1 / c) * r if c != 0 else 0
    rb = -(1 / b) * r if b != 0 else 0
    rd = (1 / d) * r if d != 0 else 0

    # Solo calcular reacciones 2 y 3 si están activas
    if c2 != 0:
        k2 = A2 * np.exp(-E2 / R / (T + 460))
        r2 = k2 * np.power(Cc, ordenc2)
        rc2 = -(1 / c2) * r2
        rz = (1 / z2) * r2 if z2 != 0 else 0
    else:
        rc2 = rz = 0

    if a3 != 0:
        k3 = A3 * np.exp(-E3 / R / (T + 460))
        r3 = k3 * np.power(Ca, ordena3)
        ra3 = -(1 / a3) * r3
        rg = (1 / g3) * r3 if g3 != 0 else 0
    else:
        ra3 = rg = 0

    # Sumar tasas de reacción totales
    ra = ra1 + ra3
    rc = rc1 + rc2

    # Balance de energía
    ThetaCp = CPa + (Fb0 / Fa0) * CPb + (Fm0 / Fa0) * CPm
    Ta2 = T - ((T - Ta1) * np.exp(-UA / (CpW * mc)))
    Qr2 = mc * CpW * (Ta2 - Ta1)
    Qr1 = Fa0 * ThetaCp * (T - T0)
    Qr = Qr1 + Qr2
    Qg = (ra1 * V * dH) + (rc2 * V * dH2) + (ra3 * V * dH3) 

    p1.set_ydata(Ca)
    p1c.set_ydata(Cc)
    p1z.set_ydata(Cz)
    p1g.set_ydata(Cg)
    p2.set_ydata(T)
    p3.set_ydata(X)
    p3s.set_ydata(S)
    p4.set_ydata(Qg)
    p5.set_ydata(Qr)
    fig.canvas.draw_idle()

sUA.on_changed(update_plot2)
sA.on_changed(update_plot2)
sE.on_changed(update_plot2)
sTa1.on_changed(update_plot2)
sCpW.on_changed(update_plot2)
sT0.on_changed(update_plot2)
smc.on_changed(update_plot2)
sdH.on_changed(update_plot2)
sV.on_changed(update_plot2)
sTi.on_changed(update_plot2)
sFa0.on_changed(update_plot2)
sFb0.on_changed(update_plot2)
sFm0.on_changed(update_plot2)
sCai.on_changed(update_plot2)
sCbi.on_changed(update_plot2)
sCmi.on_changed(update_plot2)
sroa.on_changed(update_plot2)
srob.on_changed(update_plot2)
srom.on_changed(update_plot2)
sordena.on_changed(update_plot2)
sordenb.on_changed(update_plot2)
sa.on_changed(update_plot2)
sb.on_changed(update_plot2)
sc.on_changed(update_plot2)
sd.on_changed(update_plot2)


if reaccion_2:
    sA2.on_changed(update_plot2)
    sE2.on_changed(update_plot2)
    sdH2.on_changed(update_plot2)
    sc2.on_changed(update_plot2)
    sz2.on_changed(update_plot2)
    sordenc2.on_changed(update_plot2)

if reaccion_3:  
    sA3.on_changed(update_plot2)
    sE3.on_changed(update_plot2)
    sa3.on_changed(update_plot2)
    sg3.on_changed(update_plot2)
    sdH3.on_changed(update_plot2) 
    sordena3.on_changed(update_plot2)


resetax = plt.axes([0.65, 0.96, 0.11, 0.04])
button = Button(resetax, 'Reset variables', color='cornflowerblue', hovercolor='0.975')

def reset(event):
    sUA.reset()
    sA.reset()
    sE.reset()
    sTa1.reset()
    sCpW.reset()
    sT0.reset()
    smc.reset()
    sdH.reset()
    sV.reset()
    sFa0.reset()
    sFb0.reset()
    sFm0.reset()
    sTi.reset()
    sCai.reset()
    sCbi.reset()
    sCmi.reset()
    sroa.reset()
    srob.reset()
    srom.reset()
    sordena.reset()
    sordenb.reset()
    sa.reset()
    sb.reset()
    sc.reset()
    sd.reset()

    if reaccion_2:
        sA2.reset()
        sE2.reset()
        sdH2.reset()
        sc2.reset()
        sz2.reset()
        sordenc2.reset()

    if reaccion_3:
        sA3.reset()
        sE3.reset()
        sdH3.reset()
        sa3.reset()
        sg3.reset()
        sordena3.reset()
    
    
button.on_clicked(reset)
plt.show()
    

# Crear un DataFrame con los valores que se grafican
df = pd.DataFrame({
    'time': tspan,       # Eje X (tiempo)
    'Ca': Ca,            # Concentración de A
    'Cb': Cb,            # Concentración de B
    'Cc': Cc,            # Concentración de C
    'Cm': Cm,            # Concentración de M (inerte)
    'T': T,              # Temperatura
    'Cd': Cd,            # Concentración de D
    'Cz': Cz,            # Concentración de Z
    'Cg': Cg,            # Concentración de G
    'X': X,              # Conversión
    'S': S,              # Selectividad
    'Qg': Qg,            # Calor generado
    'Qr': Qr,            # Calor removido
    'tau': tau           #t. de residencia
})

import time
output_folder = r'C:\Users\facio\Desktop\Trabajo Final\resultados de la simulacion'
filename = f'resultados_simulacion_CASOB_{time.strftime("%Y%m%d-%H%M%S")}.csv'
output_file = os.path.join(output_folder, filename)
df.to_csv(output_file, index=False)

output_file_excel = os.path.join(output_folder, f'{filename}.xlsx')
df.to_excel(output_file_excel, index=False, engine='openpyxl')