import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

df = pd.read_csv('lbm_horario_completo.dat',sep=',')
df.index = pd.to_datetime(df['TIMESTAMP'])
df.index.name = None
df.drop(columns=['TIMESTAMP','CNR1TC'],inplace=True)

df['year'] = df.index.year
df['month'] = df.index.month
df['day'] = df.index.day
df['hour'] = df.index.hour
df['Td'] = np.nan

df['Sw_dw'] = pd.concat([df['2016':'2018-10']['PSP1_Wm2'].rename({'PSP1_Wm2':'Sw_dw'}), df['2018-11':'2020']['CM3Up_Wm2'].rename({'CM3Up_Wm2':'Sw_dw'})])
df['Sw_dif'] = pd.concat([df['2018-09':'2019-08']['PSP1_Wm2'].rename({'PSP1_Wm2':'Sw_dif'}),df['2019-10':'2020']['CMP21_Wm2'].rename({'CMP21_Wm2':'Sw_dif'})])

df['u'] = pd.concat([df[:'2019-03-21']['u_WVT'].rename({'u_WVT':'u'}),df['2019-03-22':]['u_WXT'].rename({'u_WXT':'u'})])
df['v'] = pd.concat([df[:'2019-03-21']['v_WVT'].rename({'v_WVT':'v'}),df['2019-03-22':]['v_WXT'].rename({'v_WXT':'v'})])

df['WS'] = pd.concat([df[:'2019-04-20']['WS_WVT'].rename({'WS_WVT':'WS'}),df['2019-04-21':]['WS_WXT'].rename({'WS_WXT':'WS'})])
df['WD'] = pd.concat([df[:'2019-04-20']['WindDir_WVT'].rename({'WindDir_WVT':'WD'}),df['2019-04-21':]['WD_WXT'].rename({'WD_WXT':'WD'})])

df['Lw_dw'] = pd.concat([df[:'2018-07']['PIR1_Wm2'].rename({'PIR1_Wm2':'Lw_dw'}),df['2018-08':]['CG3Up_Wm2Cr'].rename({'CG3Up_Wm2Cr':'Lw_dw'})])


df.rename(columns={'PAR_Wm2':'Sw_par','CM3Dn_Wm2':'Sw_up','CG3Dn_Wm2Cr':'Lw_up','Net_Wm2':'Net_CNR1',
                    'NRLite_Wm2Cr':'Net_NRLite','Temp1':'T','RH1':'ur'},inplace=True)

df.drop(columns=['PSP1_Wm2','CM3Up_Wm2','CMP21_Wm2','u_WVT','v_WVT','u_WXT','v_WXT','CG3Up_Wm2Cr','PIR1_Wm2'],inplace=True)


teo = pd.read_csv('arquivo_radiação_teórica_labmim.csv',sep=';')
teo.index = pd.to_datetime(teo['year month day hour minute'.split()])
teo.drop(columns='lon lat alt ano_r n_dia_r'.split(),inplace=True)
teo = teo[df.index[0]:df.index[-1]]

df = pd.merge_ordered(teo,df)

df = df['year;month;day;hour;T;Td;ur;pressure;WD;WS;u;v;precip;Sw_dw;Sw_up;Lw_dw;Lw_up;Sw_par;Sw_dif;Temp_WXT;RH_WXT;Net_CNR1;Net_NRLite;oc_topo;decl_rad;elev_solar;ang_hor;ang_zen;ang_hor_por;nascer_h;por_h;ast_h;fc'.split(';')]

df.to_csv('labmim_horario_final.dat',index=False,sep=',',na_rep='nan',float_format='%.3f')