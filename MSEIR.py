#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.integrate import simps

from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime
from datetime import timedelta

class MSEIR:
    
    def __init__(self, R_0, tLat, tInf, tHosp, pMild, pFatal, pOvf, Q,
                 N, S0, E0, I0, H0, R0, D0, n_periods = 365, dt=0.1, rInf=[0], rU=[0]):
        
        self.n_periods, self.dt = n_periods, dt
        self.R_0, self.tLat, self.tInf, self.tHosp = R_0, tLat, tInf, tHosp
        self.pMild, self.pFatal, self.pOvf, self.Q  = pMild, pFatal, pOvf, Q
        self.N, self.S0, self.E0, self.I0, self.H0, self.R0, self.D0 = N, S0, E0, I0, H0, R0, D0
        
        self.rInf = np.concatenate([i*np.ones(int(1/self.dt)) for i in rInf]).ravel()
        self.rU = np.concatenate([i*np.ones(int(1/self.dt)) for i in rU]).ravel()
        self.Beta =  self.R_0/self.tInf
        self.t = int(self.n_periods/self.dt)
    
    def _ovf(self, H):
        
        return self.pFatal if (round(H,0) <= self.Q) else self.pOvf
 
    def _ode(self, Ut, pS, pE, pI, pH, pR, pD):
        
        nS = pS - self.dt*((1 - Ut)*self.Beta*((pS*pI)/self.N))
        nE = pE + self.dt*((1 - Ut)*self.Beta*((pS*pI)/self.N) - pE/self.tLat)
        nI = pI + self.dt*(pE/self.tLat - pI/self.tInf)
        nH = pH + self.dt*((1 - self.pMild)*pI/self.tInf - pH/self.tHosp)
        nR = pR + self.dt*(self.pMild*pI/self.tInf + (1-self._ovf(pH))*pH/self.tHosp)
        nD = pD + self.dt*(self._ovf(pH)*pH/self.tHosp)
        
        return [nS, nE, nI, nH, nR, nD]
    
    def _costf(self, x, S0, E0, I0, H0, R0, D0, hor):

        C = [[S0, E0, I0, H0, R0, D0]]
        t = int(hor/self.dt)
        Uf = x * np.ones(t)

        for i in range(t):

            Ut = Uf[i]
            nVal = self._ode(Ut, C[-1][0], C[-1][1], C[-1][2], C[-1][3], C[-1][4], C[-1][5])
            C.append(nVal)

        solution = np.asarray(C)
        i = solution[:, 3] - self.Q
        cost_func = simps((i+abs(i))/2) + simps(Uf)

        return cost_func
    
    def solve(self, U=0, optimize=False, solver='SLSQP', bounds=(0.0, 0.8), freq=1, hor=10):
        
        C = [[self.S0, self.E0, self.I0, self.H0, self.R0, self.D0]]
        y = self.rU if (len(self.rU)>1/self.dt) else [float(U)]
        Uf = np.concatenate([y, [float(U)] * self.t]).ravel()

        for i in range(self.t):

            Ut = Uf[i]
            nVal = self._ode(Ut, C[-1][0], C[-1][1], C[-1][2], C[-1][3], C[-1][4], C[-1][5])
            C.append(nVal)
            
            if optimize and i>=len(self.rU) and (i%int(freq/self.dt) == 0):
                
                args = tuple(nVal + [hor])
                optimal_Ut = minimize(self._costf, x0 = Ut, args = args, method = solver, bounds=(bounds,))
                Uf[i:] = optimal_Ut.x.round(10) * np.ones(len(Uf[i:]))
            
        solution = np.asarray(C)
        df_names = ['S', 'E', 'I', 'H', 'R', 'D']
        df_res = pd.DataFrame(solution, columns=df_names)
        df_res['t'] = range(self.t+1)
        df_res['Q'] = self.Q
        df_res['Uf'] = Uf[:len(df_res)]
        df_res['mInf'] = -df_res['S'].diff()
        df_res['rInf'] = pd.Series(self.rInf).diff(periods=1/self.dt)

        return df_res
    
    def plot(self, data, since='2020-01-01', size=(700,900), comps='SEIHRD', title='SEIHRD model with optimal control'):
        
        xT = pd.date_range(start = since,
                           end = datetime.strptime(since, "%Y-%m-%d") + timedelta(days=self.n_periods),
                           periods=self.n_periods).strftime("%Y-%m-%d")

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            horizontal_spacing=0.01,
                            vertical_spacing=0.01,
                            row_heights=[0.15, 0.70, 0.15])
        
        colors = ['#910d00', '#086d71', '#086d71', '#729fcf', '#3465a4',
                  '#086d71', '#cc397b', '#9ca4c4', '#380915', '#1e90ff']
        
        lines = [dict(color=x, width=1.5) for x in colors]
    
        fig.add_trace(go.Scatter(x=xT, y=data['Uf'], opacity=1.0,name='Uf', line=lines[0]), row=1, col=1)
        
        if 'S' in comps:
            fig.add_trace(go.Scatter(x=xT, y=data['S'], opacity=1.0,name='S', line=lines[3]), row=2, col=1)
        if 'E' in comps:
            fig.add_trace(go.Scatter(x=xT, y=data['E'], opacity=0.8,name='E', line=lines[4]), row=2, col=1)
        if 'I' in comps:
            fig.add_trace(go.Scatter(x=xT, y=data['I'], opacity=0.8,name='I', line=lines[5]), row=2, col=1)
        if 'H' in comps:
            fig.add_trace(go.Scatter(x=xT, y=data['H'], opacity=0.8,name='H', line=lines[6]), row=2, col=1)
        if 'R' in comps:
            fig.add_trace(go.Scatter(x=xT, y=data['R'], opacity=1.0,name='R', line=lines[7]), row=2, col=1)  
        if 'D' in comps:
            fig.add_trace(go.Scatter(x=xT, y=data['D'], opacity=0.8,name='D', line=lines[8]), row=2, col=1)
        
        fig.add_trace(go.Scatter(x=xT, y=data['Q'], opacity=1.0, name='Q',
                                 line=dict(color='#cc397b', width=1.5, dash='dot')), row=2, col=1)
        
        fig.add_trace(go.Scatter(x=xT, y=data['mInf'], opacity=0.3,name='mInf', line=lines[1]), row=3, col=1)
        fig.add_trace(go.Scatter(x=xT, y=data['rInf'], opacity=0.6,name='rInf', line=lines[2]), row=3, col=1)
        
        fig.update_yaxes(title_text="u(t)", row=1, col=1, nticks=3, showgrid=False)
        fig.update_yaxes(title_text="SEIHRD, Q", row=2, col=1, nticks=3,showgrid=False)
        fig.update_yaxes(title_text="Inf(t)", row=3, col=1, nticks=3, showgrid=False)
        
        fig.update_layout(height=size[0], width=size[1],
                          legend_orientation="h", legend={'x':0, 'y':1.06, 'itemsizing': 'constant'},
                          title_text=title)
        return fig
