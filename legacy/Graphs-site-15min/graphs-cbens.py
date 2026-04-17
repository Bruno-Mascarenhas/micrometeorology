
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

labmim = pd.read_csv('rad_labmim_bruno.dat',sep=';')
labmim.index = pd.to_datetime(labmim[['year', 'month', 'day', 'hour']])

sw_dir = []
for dw, dif in zip(labmim['Sw_dw'],labmim['Sw_dif']):
    if not np.isnan(dw) and not np.isnan(dif) and dw-dif > 0:
        sw_dir.append(dw-dif)
    else:
        sw_dir.append(np.nan)

labmim['Sw_dir'] = sw_dir

#panorama completo
labmim['Sw_top'].plot()
labmim['Sw_dw'].plot()
#labmim['Sw_dir'].plot()
labmim['Sw_dif'].plot()
plt.ylabel('Radiação (W/m²)')
plt.legend()
plt.savefig('rad-panorama.png',bbox_inches='tight')
#plt.show()
plt.close()

###############################################################
#daily######################################################## VERAO
#topo
x = [x+1 for x in range(24)]
mm = labmim.groupby(['month', 'hour'])['Sw_top']
line = []
for i in range(24):
        line.append(mm.get_group((1,i)).dropna().mean())
line = plt.plot(x,np.array(line)*0.0036,color='black')

#dif
mm = labmim.groupby(['month', 'hour'])['Sw_dif']
bp1 = []
for i in range(24):
    bp1.append(list(mm.get_group((1,i)).dropna().values*0.0036))
for i in range(24):
    bp1[i] += list(mm.get_group((2,i)).dropna().values*0.0036)
for i in range(24):
    bp1[i] += list(mm.get_group((12,i)).dropna().values*0.0036)


#direta
mm = labmim.groupby(['month', 'hour'])['Sw_dir']
bp2 = []
for i in range(24):
    bp2.append(list(mm.get_group((1,i)).dropna().values*0.0036))
for i in range(24):
    bp2[i] += list(mm.get_group((2,i)).dropna().values*0.0036)
for i in range(24):
    bp2[i] += list(mm.get_group((12,i)).dropna().values*0.0036)


#global
mm = labmim.groupby(['month', 'hour'])['Sw_dw']
bp3 = []
for i in range(24):
    bp3.append(list(mm.get_group((1,i)).dropna().values*0.0036))
for i in range(24):
    bp3[i] += list(mm.get_group((2,i)).dropna().values*0.0036)
for i in range(24):
    bp3[i] += list(mm.get_group((12,i)).dropna().values*0.0036)

props_dw = plt.boxplot(bp3,labels=np.arange(0,24,1),sym='',patch_artist=True)
#props_dir = plt.boxplot(bp2,labels=np.arange(0,24,1),sym='',patch_artist=True)
props_dif = plt.boxplot(bp1,labels=np.arange(0,24,1),sym='',patch_artist=True)

for box in props_dw['boxes']:
    box.set(facecolor='red',alpha=0.5)
#for box in props_dir['boxes']:
#    box.set(facecolor='blue')
for box in props_dif['boxes']:
    box.set(facecolor='blue',alpha=0.5)

plt.legend([line[0],props_dw['boxes'][0],props_dif['boxes'][0]],['Sw_top','Sw_dw','Sw_dif'])
plt.ylabel('Radiação (MJ/m²h)')
plt.xlabel('Horas')
plt.title('Verão (DJF)')
plt.xlim(5,20)
plt.ylim(-0.5,6)

plt.savefig('rad-hourly-verao.png',bbox_inches='tight')
#plt.show()
plt.close()

#daily######################################################## INVERNO
#topo
x = [x+1 for x in range(24)]
mm = labmim.groupby(['month', 'hour'])['Sw_top']
line = []
for i in range(24):
        line.append(mm.get_group((7,i)).dropna().mean())
line = plt.plot(x,np.array(line)*0.0036,color='black')

#dif
mm = labmim.groupby(['month', 'hour'])['Sw_dif']
bp1 = []
for i in range(24):
    bp1.append(list(mm.get_group((6,i)).dropna().values*0.0036))
for i in range(24):
    bp1[i] += list(mm.get_group((7,i)).dropna().values*0.0036)
for i in range(24):
    bp1[i] += list(mm.get_group((8,i)).dropna().values*0.0036)

#direta
mm = labmim.groupby(['month', 'hour'])['Sw_dir']
bp2 = []
for i in range(24):
    bp2.append(list(mm.get_group((7,i)).dropna().values*0.0036))
for i in range(24):
    bp2[i] += list(mm.get_group((7,i)).dropna().values*0.0036)
for i in range(24):
    bp2[i] += list(mm.get_group((8,i)).dropna().values*0.0036)


#global
mm = labmim.groupby(['month', 'hour'])['Sw_dw']
bp3 = []
for i in range(24):
    bp3.append(list(mm.get_group((7,i)).dropna().values*0.0036))
for i in range(24):
    bp3[i] += list(mm.get_group((7,i)).dropna().values*0.0036)
for i in range(24):
    bp3[i] += list(mm.get_group((8,i)).dropna().values*0.0036)

props_dw = plt.boxplot(bp3,labels=np.arange(0,24,1),sym='',patch_artist=True)
#props_dir = plt.boxplot(bp2,labels=np.arange(0,24,1),sym='',patch_artist=True)
props_dif = plt.boxplot(bp1,labels=np.arange(0,24,1),sym='',patch_artist=True)

for box in props_dw['boxes']:
    box.set(facecolor='red',alpha=0.5)
#for box in props_dir['boxes']:
#    box.set(facecolor='blue')
for box in props_dif['boxes']:
    box.set(facecolor='blue',alpha=0.5)

plt.legend([line[0],props_dw['boxes'][0],props_dif['boxes'][0]],['Sw_top','Sw_dw','Sw_dif'])
plt.ylabel('Radiação (MJ/m²h)')
plt.xlabel('Horas')
plt.title('Inverno (JJA)')
plt.xlim(5,20)
plt.ylim(-0.5,6)

plt.savefig('rad-hourly-inverno.png',bbox_inches='tight')
#plt.show()
plt.close()

####################################################
#mensal
######################################################

top = labmim.loc[labmim['Sw_top']>1].groupby(['year', 'month', 'day']).sum().reset_index().groupby('month').mean()['Sw_top']
x = [x for x in range(1,13)]
plt.plot(x,top*0.0036,color='black')

sw = labmim.groupby(['year', 'month', 'day']).sum()['Sw_dw'].reset_index().groupby('month')['Sw_dw']
dif = labmim.loc[labmim['Sw_dif']>1].groupby(['year', 'month', 'day']).sum()['Sw_dif'].reset_index().groupby('month')['Sw_dif']
dir = labmim.loc[labmim['Sw_dir']>1].groupby(['year', 'month', 'day']).sum()['Sw_dir'].reset_index().groupby('month')['Sw_dir']

bp1 = []
for i in range(1,13):
    try:
        tmp = []
        for x in sw.get_group( i ).dropna()*0.0036:
            if x > 0:
                tmp.append(x)
        bp1.append(np.array(tmp))
    except:
        bp1.append([])

bp2 = []
for i in range(1,13):
    try:
        bp2.append(dif.get_group( i ).dropna()*0.0036)
    except:
        bp2.append([])

bp3 = []
for i in range(1,13):
    try:
        bp3.append(dir.get_group( i ).dropna()*0.0036)
    except:
        bp3.append([])

props_dw = plt.boxplot(bp1,labels=np.arange(1,13,1),sym='',patch_artist=True)
#props_dir = plt.boxplot(bp3,labels=np.arange(1,13,1),sym='',patch_artist=True)
props_dif = plt.boxplot(bp2,labels=np.arange(1,13,1),sym='',patch_artist=True)

for box in props_dw['boxes']:
    box.set(facecolor='red',alpha=0.5)
#for box in props_dir['boxes']:
#    box.set(facecolor='blue')
for box in props_dif['boxes']:
    box.set(facecolor='blue',alpha=0.5)

plt.legend([line[0],props_dw['boxes'][0],props_dif['boxes'][0]],['Sw_top','Sw_dw','Sw_dif'])
plt.ylabel('Radiação (MJ/m²h)')
plt.xticks(range(1,13),['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])

plt.savefig('rad-monthly.png',bbox_inches='tight')
#plt.show()
plt.close()
