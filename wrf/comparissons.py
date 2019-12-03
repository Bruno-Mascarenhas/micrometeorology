
import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sys import exit

#class to simplify paring of two datasets
class PairDf():
    #atributes
    def __init__(self,obs,pred):
        self.dataframes = dict()
        PairDf.fit(self,obs,pred)
    
    #method to paring over columns and return a dict with paired dataframes for each column in commom
    def fit(self,obs,pred):
        idx1 = obs.index.intersection(pred.index)
        obs = obs.loc[idx1,:]
        idx2 = pred.index.intersection(obs.index)
        pred = pred.loc[idx2,:]

        for col in obs.columns:
            if col in pred.columns:
                df = pd.DataFrame(data=obs[col])
                df.rename(columns={col:'obs'},inplace=True)
                df['pred'] = pred[col]
                df.dropna(inplace=True)
                self.dataframes.update({col:df})

    #method to return a dict with all metrics
    def dict_metrics(self,var):
        ans = dict()
        #put here the functions
        functions = [metrics.r,metrics.MBD,metrics.RMSD]
        for func in functions:
            ans[func.__name__]=func(self.dataframes[var]['obs'],self.dataframes[var]['pred'])
        return ans

    def get_paired(self):
        return self.dataframes

def generate_metrics(files,foutput,name,variables,months):
    """
    files = Tuple/list with absolute location of files and name of each station in third coord. Eg. ('/home/est1.dat','/home/model1.dat','est1')
    foutput = Path to save the excel readable metrics
    name = name of the file
    variables = desired variables for compare
    months = months to compare
    """
    total_metric = dict()
    hourly_metric = dict()

    for file in files:
        station = file[2]
        try:
            obs = pd.read_csv(file[0],sep=',')
            obs.index = pd.to_datetime(obs['year month day hour'.split()])
            obs = obs[~obs.index.duplicated()]

            pred = pd.read_csv(file[1],sep=',')
            pred.index = pd.to_datetime(pred['year month day hour'.split()])
            pred = pred[~pred.index.duplicated()]
            pred['WS'] = np.sqrt(pred['u']**2+pred['v']**2)
            pred['q'] = pred['q']/1000
            pred['pressure'] = pred['pressure']
            pred['e'] = (pred['q']*pred['pressure'])/(0.622*(1-pred['q'])+pred['q'])
            pred['es'] = 611.2*np.exp((17.67*pred['T'])/(pred['T']+243.5))
            pred['ur'] = (pred['e']/pred['es'])*100
            pred['q'] *= 1000
            pred['pressure'] /= 100
            pred['e'] /= 100

            if len(months) > 1:
                pred = pd.concat([pred.loc[pred['month']==month,:] for month in months])
                obs = pd.concat([obs.loc[obs['month']==month,:] for month in months])
            else:
                pred = pred.loc[pred['month']==months[0],:]
                obs = obs.loc[obs['month']==months[0],:]
          
            #choosen variables
            for var in variables:
                try:
                    if var == 'H':
                        pred.loc[pred[var]<=-100,var] = np.nan
                    else:
                        pred.loc[pred[var]<=0,var] = np.nan
                        obs.loc[obs[var]<=0,var] = np.nan
                except:
                    continue
            
            #fit to total df
            compare = PairDf(obs.drop(columns='year month day hour'.split()),pred.drop(columns='year month day hour'.split()))
            
            total_metric[station] = {}

            for var in variables:
                try:
                    x = compare.dict_metrics(var)
                    total_metric[station][var]=x
                except:
                    continue

            #fit for hourly metrics
            obsh = obs.groupby('hour')
            predh = pred.groupby('hour')

            hourly_metric[station] = {}

            for i in range(24):
                hourly_metric[station][i] = {}

                compare = PairDf(obsh.get_group(i).drop(columns='year month day hour'.split()),predh.get_group(i).drop(columns='year month day hour'.split()))

                for var in variables:
                    try:
                        x = compare.dict_metrics(var)
                        hourly_metric[station][i][var]=x
                    except:
                        continue

        except Exception as err:
                print(file,err)

    dft = pd.concat({k:pd.DataFrame(v) for k,v in total_metric.items()},sort=True).unstack()
    dft.to_excel(foutput+name+'total.xlsx')

    dfh = pd.concat({x:pd.concat({k:pd.DataFrame(v) for k,v in hours.items()},sort=True) for x,hours in hourly_metric.items()},sort=True).unstack()
    dfh.to_excel(foutput+name+'hourly.xlsx')

def generate_mean(files,foutput,name,variables,months):
    """
    files = Tuple/list with absolute location of files and name of each station in third coord. Eg. ('/home/est1.dat','/home/model1.dat','est1')
    foutput = Path to save the excel readable means
    name = name of the file
    variables = desired variables for compare
    months = months to compare
    """
    monthly_mean_obs = dict()
    monthly_mean_pred = dict()

    for file in files:
        station = file[2]
        try:
            obs = pd.read_csv(file[0],sep=',')
            obs.index = pd.to_datetime(obs['year month day hour'.split()])
            obs = obs[~obs.index.duplicated()]

            pred = pd.read_csv(file[1],sep=',')
            pred.index = pd.to_datetime(pred['year month day hour'.split()])
            pred = pred[~pred.index.duplicated()]
            pred['WS'] = np.sqrt(pred['u']**2+pred['v']**2)
            pred['q'] = pred['q']/1000
            pred['pressure'] = pred['pressure']
            pred['e'] = (pred['q']*pred['pressure'])/(0.622*(1-pred['q'])+pred['q'])
            pred['es'] = 611.2*np.exp((17.67*pred['T'])/(pred['T']+243.5))
            pred['ur'] = (pred['e']/pred['es'])*100
            pred['q'] *= 1000
            pred['pressure'] /= 100
            pred['e'] /= 100

            for var in variables:
                try:
                    if var == 'H':
                        pred.loc[pred[var]<=-100,var] = np.nan
                    else:
                        pred.loc[pred[var]<=0,var] = np.nan
                        obs.loc[obs[var]<=0,var] = np.nan
                except:
                    continue

            monthly_mean_obs[station] = {}
            monthly_mean_pred[station] = {}

            for i in months:
                monthly_mean_obs[station][i] = {}
                monthly_mean_pred[station][i] = {}
                for var in variables:
                    try:
                        x = obs.loc[obs['month'] == i,var].mean()
                        monthly_mean_obs[station][i][var] = x                    
                        x = pred.loc[pred['month'] == i,var].mean()
                        monthly_mean_pred[station][i][var] = x
                    except:
                        continue
        except Exception as err:
            print(file,err)

    dft = pd.concat({k:pd.DataFrame(v) for k,v in monthly_mean_pred.items()},sort=True)
    dft.to_excel(foutput+name+'_means_pred.xlsx')

    dft = pd.concat({k:pd.DataFrame(v) for k,v in monthly_mean_obs.items()},sort=True)
    dft.to_excel(foutput+name+'_means_obs.xlsx')


def generate_distributions(files,foutput,name,variables,months):
    """
    files = Tuple/list with absolute location of files and name of each station in third coord. Eg. ('/home/est1.dat','/home/model1.dat','est1')
    foutput = Path to save the distribution.jpg
    name = name of the file
    variables = desired variables for compare
    months = months to compare
    """

    for file in files:
        station = file[2]
        try:
            obs = pd.read_csv(file[0],sep=',')
            obs.index = pd.to_datetime(obs['year month day hour'.split()])
            obs = obs[~obs.index.duplicated()]

            pred = pd.read_csv(file[1],sep=',')
            pred.index = pd.to_datetime(pred['year month day hour'.split()])
            pred = pred[~pred.index.duplicated()]
            pred['WS'] = np.sqrt(pred['u']**2+pred['v']**2)
            pred['q'] = pred['q']/1000
            pred['pressure'] = pred['pressure']
            pred['e'] = (pred['q']*pred['pressure'])/(0.622*(1-pred['q'])+pred['q'])
            pred['es'] = 611.2*np.exp((17.67*pred['T'])/(pred['T']+243.5))
            pred['ur'] = (pred['e']/pred['es'])*100
            pred['q'] *= 1000
            pred['pressure'] /= 100
            pred['e'] /= 100
             

            if len(months) > 1:
                pred = pd.concat([pred.loc[pred['month']==month,:] for month in months])
                obs = pd.concat([obs.loc[obs['month']==month,:] for month in months])
            else:
                pred = pred.loc[pred['month']==months[0],:]
                obs = obs.loc[obs['month']==months[0],:]

            for var in variables:
                try:
                    if var == 'H':
                        pred.loc[pred[var]<=-100,var] = np.nan
                    else:
                        pred.loc[pred[var]<=0,var] = np.nan
                        obs.loc[obs[var]<=0,var] = np.nan
                except:
                    continue
            
            #fit to total df
            compare = PairDf(obs.drop(columns='year month day hour'.split()),pred.drop(columns='year month day hour'.split()))

            pairs = compare.get_paired()

            bins = {'WS':0.5,'Sw_dw':50,'ur':5,'T':1,'q':1,'ustar':0.1}
            unidade = {'WS':'U (m/s)','Sw_dw ':'Sw_dw (W/m²)','ur':'UR (%)','T':'T (°C)','q':'Q (g/Kg)','ustar':'USTAR','pressure':'hPa'}

            sns.set(font_scale=1.3)
            sns.set_style('white')
            
            #print(pairs.keys())

            for var in pairs.keys():
                fig, ax1 = plt.subplots(nrows=1,ncols=1,sharex=True,sharey=True,figsize=(10,6))
                ax11 = ax1.twinx()

                #fig.suptitle('Distributions')
                
                if var in bins.keys():
                    #relative
                    mn = min(pairs[var]['obs'].min(),pairs[var]['pred'].min())
                    mx = max(pairs[var]['pred'].max(),pairs[var]['obs'].max())
                    b = (mx - mn)/bins[var]
                    
                    #relativess
                    sns.distplot(pairs[var]['obs'],kde=False,bins=int(b),hist_kws={'edgecolor':'black','facecolor':'gray'},norm_hist=True,ax=ax1)
                    sns.distplot(pairs[var]['pred'],kde=True,hist=False,kde_kws={'linestyle':'dashed','color':'blue'},ax=ax1)
                    #cumulative
                    sns.distplot(pairs[var]['obs'],kde=True,hist=False,kde_kws={'cumulative':True},color='black',ax=ax11)
                    sns.distplot(pairs[var]['pred'],kde=True,hist=False,kde_kws={'cumulative':True,'linestyle':'dashdot'},color='red',ax=ax11)

                    plt.xlim(mn,mx)

                else:
                    mn = min(pairs[var]['obs'].min(),pairs[var]['pred'].min())
                    mx = max(pairs[var]['pred'].max(),pairs[var]['obs'].max())

                    #relative
                    sns.distplot(pairs[var]['obs'],kde=False,hist_kws={'edgecolor':'black','facecolor':'gray'},norm_hist=True,ax=ax1)
                    sns.distplot(pairs[var]['pred'],kde=True,hist=False,kde_kws={'linestyle':'dashed','color':'blue'},ax=ax1)

                    #cumulative
                    sns.distplot(pairs[var]['obs'],kde=True,hist=False,kde_kws={'cumulative':True},color='black',ax=ax11)
                    sns.distplot(pairs[var]['pred'],kde=True,hist=False,kde_kws={'cumulative':True,'linestyle':'dashdot'},color='red',ax=ax11)

                    plt.xlim(mn,mx)


                ax1.set_xlabel(unidade[var], fontsize=14)
                ax1.set_ylabel('Relative', fontsize=14)
                ax11.set_ylabel('Cumulative', fontsize=14)



                plt.savefig(foutput+station+'-'+var+'-'+name+'.png',bbox_inches='tight',quality=100,format='png')
                plt.close()

        except Exception as err:
            print(file,err)

