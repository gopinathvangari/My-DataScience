
# python eda class

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
from scipy import stats
import numpy as np
from IPython.display import display
from scipy.stats import chi2_contingency
from scipy.stats import spearmanr
import pandas as pd
import numpy as np
import math




def gini_one(target_dict):
    
    target_dict_vals =list(target_dict.values())
    target_dict_prob =[round(x/sum(target_dict_vals),3) for x in target_dict_vals]
    base_gini = round(1 - sum([x**2 for x in target_dict_prob  ]),3)
    return base_gini
    
def entropy_one(target_dict,base='bits'):
    
    if base =='bits':
        base_log =2
    elif base=="dits":
        base_log =10
    else:
        base_log=np.e
        
    target_dict_vals =list(target_dict.values())
    target_dict_prob =[round(x/sum(target_dict_vals),3) for x in target_dict_vals]
    
    base_entropy =round(np.sum([x*math.log(1/x,base_log) for x in target_dict_prob]),3)
    return base_entropy

def ginivar(data,ftr,target):
    
    target_dict=data[target].value_counts().to_dict()
    base_gini =gini_one(target_dict)
    
    data_dict =data[ftr].value_counts().to_dict()
    data_dict_prob ={}
    for k,v in data_dict.items():
        data_dict_prob[k]=round(v/np.sum(list(target_dict.items())),3)
    
    final_gini =base_gini
    for key in data_dict.keys():
        data_now =data[(data[ftr]==key)]
        data_now_gini =gini_one(data_now[target].value_counts().to_dict())
        final_gini -=data_dict_prob[k]*data_now_gini
        
    return final_gini
    
def mutual_info(data,ftr,target,munit='bits'):
    
    #d=pd.crosstab(data_small['X'],data_small['Survived']).values
    #mi = mutual_info_score(None, None, contingency=d)
    #from sklearn.metrics import mutual_info_score
    
    target_dict=data[target].value_counts().to_dict()
    base_entropy=entropy_one(target_dict,base=munit)
    
    data_dict =data[ftr].value_counts().to_dict()
    data_dict_prob ={}
    for k,v in data_dict.items():
        data_dict_prob[k]=round(v/np.sum(list(target_dict.items())),3)
    #print(data_dict_prob)    
        
    cond_entropy=0.0
    for key in data_dict.keys():
        #print(key)
        data_now =data[(data[ftr]==key)]
        data_now_entropy =entropy_one(data_now[target].value_counts().to_dict(),base=munit)
        #print(data_now_entropy)
        cond_entropy +=data_dict_prob[key]*data_now_entropy
        
    
    mutaul_info =round(base_entropy-cond_entropy,3)
    
    
    return mutaul_info

def custom_sum(series):
    from functools import reduce
    # adding one to avoid zero dvision error
    return reduce(lambda x,y: x+y,series)+1

def custom_count(series):
    # adding one to avoid zero dvision and balance the addition of custom sum
    return len(series)+1

def iv_cat(x):
    if x < 0.02:
        return "IGNORE"
    elif 0.02 <= x < 0.01:
        return "WEAK"
    elif 0.01 <= x < 0.03:
        return "MEDIUM"
    elif 0.03 <= x < 0.5:
        return 'STRONG'
    else:
        return "SUSPICIOUS"



def iv_cal_cat(data_in,target,var_name='job'):
    
    data_x=data_in[var_name]
    data = pd.DataFrame({var_name:data_x.values,'target':data_in[target].values})
    
    # This change  has been made to avoid zero dvivision error 
    ##########################################################
    data_sum = data.groupby([var_name])['target'].agg([custom_sum, custom_count]).\
             reset_index().rename(columns={'custom_count':'total','custom_sum' : 'total_target'})
    
    # This is standard process of writing the pandas code
    ######################################################################
    #data_sum = data.groupby([var_name])['target'].agg(['sum', 'count']).\
            # reset_index().rename(columns={'count':'total','sum' : 'total_target'})
    ##############################################################################    
    
    data_sum['target_rate']=data_sum['total_target']/data_sum['total']
    data_sum['non_target']=data_sum['total'] -data_sum['total_target']
    
    target =np.sum(data_sum['total_target'])
    non_target =np.sum(data_sum['non_target'])
    
    data_sum['woe'] =np.log((data_sum['total_target']/target)/(data_sum['non_target']/non_target))
    data_sum['rate_diff']=(data_sum['total_target']/target) - (data_sum['non_target']/non_target)
    
    data_sum['iv_gr']=data_sum['woe']*data_sum['rate_diff'] 
    
    iv_val =np.sum(data_sum['iv_gr'])
    
    return iv_val


def conti_mono_bin(data,target,var_name='job',n = 20):
    
    
    
    data_x=data[var_name]
    trny=data[target]
    
    data_x=data_x.fillna(np.median(data_x))
    r = 0
    while abs(r) <1 :
        if n==1:
            d1 = pd.DataFrame({var_name: data_x, "target":trny.values , "Bins": pd.qcut(data_x, 
                                                                                n,duplicates='drop',precision = 100)})
            d2 = d1.groupby('Bins', as_index = True)
            break
        else:
            d1 = pd.DataFrame({var_name: data_x, "target":trny.values , "Bins": pd.qcut(data_x, 
                                                                                 n,duplicates='drop',precision = 100)})
            d2 = d1.groupby('Bins', as_index = True)
            r,p = spearmanr(d2.mean()[var_name], d2.mean()['target'])
            n =n-1
    #print(d2.groups)  
    d3 = pd.DataFrame(d2.min()[var_name].values, columns = ['min_' + var_name])
    #print(d3)
    d3['max_' + var_name] = d2.max()[var_name].values
    #print(d3)
    d3['total_target'] = d2.sum()['target'].values
    d3['total'] = d2.count()['target'].values
    d3['target' + '_rate'] = d2.mean()['target'].values
    data_s = (d3.sort_values(by = 'min_' + var_name)).reset_index(drop = True)
    
    
    
    def adj_row(row):
        if row['total_target']==0:
            row['total_target']=1
            row['total']=row['total']+1
        return row    
    
    
    data_sum =data_s.apply(lambda row :adj_row(row),axis=1)     
    data_sum['non_target']=data_sum['total'] -data_sum['total_target']
    target =np.sum(data_sum['total_target'])
    non_target =np.sum(data_sum['non_target'])
    
    data_sum['woe'] =np.log((data_sum['total_target']/target)/(data_sum['non_target']/non_target))
    data_sum['rate_diff']=(data_sum['total_target']/target) - (data_sum['non_target']/non_target)
    data_sum['iv_gr']=data_sum['woe']*data_sum['rate_diff'] 
    
    iv_val =np.sum(data_sum['iv_gr'])
    
    data_sum[var_name]=data_sum.apply(lambda row: str(row['min_'+ var_name])+'-'+str(row['max_'+ var_name]),axis=1)
    # Add Entropy Data
    tot_records =data_sum['total'].sum()
    data_sum['entro_weight']=data_sum['total']/tot_records
    data_sum['cond_target']=data_sum.apply(lambda x: x['total_target']/x['total'],axis=1)
    data_sum['cond_non_target']=data_sum.apply(lambda x: x['non_target']/x['total'],axis=1)
    data_sum['row_entropy']=data_sum.apply(lambda x: x['entro_weight']*(x['cond_target']*np.log(x['cond_target'])+ \
                                                                         
                                                   x['cond_non_target']*np.log(x['cond_non_target'])),axis=1)
    # Add Entropy Data
    data_sum['cond_gini']=data_sum.apply(lambda x: x['entro_weight']*(1-x['cond_target']**2 -x['cond_non_target']**2),axis=1)
    base_entro =-1*((data_sum['total_target'].sum()/tot_records)*np.log(data_sum['total_target'].sum()/tot_records)+\
    (data_sum['non_target'].sum()/tot_records)*np.log(data_sum['non_target'].sum()/tot_records))
    
    mutual_info =base_entro - (-1*data_sum['row_entropy'].sum())
    
    base_gini = 1-(data_sum['total_target'].sum()/tot_records)**2 -(data_sum['non_target'].sum()/tot_records)**2
    
    gini_val = base_gini - data_sum['cond_gini'].sum()
    
    chi2, p, dof, ex =chi2_contingency(data_sum[['non_target','total_target']].values,
                                           correction=True)
    
    return data_sum,iv_val,gini_val,mutual_info,chi2, p

def grp_manager(row,ftr_name,l,u,cat_gp):
    
    res=''
    for i in range(len(l)):
        if row[ftr_name] >=l[i]-0.5 and row[ftr_name]<=u[i]+0.5:
            res =cat_gp[i]
            break
    return res



class EdaOne:
    
    '''
    EdaOne Class would be initialized Pandas dataframe, dataframe name,target feature string and path
    
    Attributes: 
        Data    : Pandas Dataframe. 
        Name    : Assigned Name for dataframe
        Target  : Target Feature String
        Path    : Output Destiantion
            
    '''
    
     
    
    def __init__(self,data,name,target,path='D:\\junk_yard\\'):
        self.data =data
        self.num_vars =[key for key in dict(data.dtypes) if dict(data.dtypes)[key] 
                        in ['float64','float32','int32','int64'] and key !=target ]
        self.target_name =target
        self.cat_vars= cat_vars = [key for key in dict(data.dtypes) if dict(data.dtypes)[key] in ['object'] ]
        self.path=path
        self.name =name
        
    def get_miss_report(self,viz=True):
        
        ff =pd.DataFrame(self.data.isnull().sum(),columns=['count_miss'])
        ff['col_names']=ff.index
        ff_2 =ff.reset_index(drop=True)
        ff_3 =ff_2[['col_names','count_miss']]
        ff_3.to_csv(self.path+self.name+'_miss_report.csv')
        if viz:
            fig,ax_1 =plt.subplots(1,1,figsize=(10,5))
            ax_1=sns.barplot(x=ff_3['col_names'].values,y=ff_3['count_miss'].values)
            ax_1.set_xlabel("Feature Names")
            ax_1.xaxis.set_tick_params(rotation=45)
            ax_1.set_ylabel("Count Missing")
            ax_2 = ax_1.twinx()
            ax_2=sns.lineplot(x=ff_3['col_names'].values,y=ff_3['count_miss'].values/self.data.shape[0],sort=False)
            ax_1.set_title("Missing Data  Report Viz",fontsize=15)
            ax_2.set_xlabel("Feature Names")            
            ax_2.set_ylabel("Percent Missing")
            ax_2.grid(False)
            
    def get_outlier_report(self,viz=False):
        
        '''
        Simple Detection of outlier for continuous variable...
        
        '''
        low_data =[]
        high_data =[]
        low_val=[]
        high_val =[]
        med_val=[]
        name_list=[]
        for item in self.num_vars:
            median_item =round(self.data[item][self.data[item].notnull()].median(),2)
            q1_item =round(self.data[item][self.data[item].notnull()].quantile(0.25),2)
            q3_item =round(self.data[item][self.data[item].notnull()].quantile(0.75),2)
            iqr_item=q3_item-q1_item
            high_item =q3_item+1.5*iqr_item
            low_item =q1_item-1.5*iqr_item
            low_nos =np.sum(self.data[item][self.data[item].notnull()] < low_item)
            high_nos=np.sum(self.data[item][self.data[item].notnull()] > high_item)
            med_val.append(median_item)
            low_val.append(q1_item)
            high_val.append(q3_item)
            low_data.append(low_nos)
            high_data.append(high_nos)
            name_list.append(item)
        data_dict ={'Feature_Name':name_list,'Median':med_val,'Q1_Value':low_val,'Q3_Value':high_val,
                   'Left_Outlier_Nos':low_data,'Right_Outlier_Nos':high_data}
                       
        
        data_outlier =pd.DataFrame(data_dict,
                                  columns=['Feature_Name','Median','Q1_Value',
                                           'Q3_Value','Left_Outlier_Nos','Right_Outlier_Nos'])
            
            
        data_outlier.to_csv(self.path+self.name+'_outlier_report.csv')
        if viz==True:
            display(data_outlier)
            
        
    def dstat_report(self):
        
        num_dstat =self.data[self.num_vars].describe()
        cat_dstat =self.data[self.cat_vars].describe()
        
        num_dstat.to_csv(self.path+self.name+'_num_dstat.csv',index=True)
        cat_dstat.to_csv(self.path+self.name+'_cat_dstat.csv',index=True)
        return num_dstat,cat_dstat
        
    def cat_one_viz(self,ftr_name='Age',munit='nats'):
        
        data_cat =self.data[[ftr_name,self.target_name]]
        target_rate =round(sum(data_cat[self.target_name])/data_cat.shape[0],2)
        #print(target_rate)
        gini_val =ginivar(data_cat,ftr_name,self.target_name)
        mi_val =mutual_info(data_cat,ftr_name,self.target_name,munit=munit)
        chi2, p, dof, ex =chi2_contingency(pd.crosstab(data_cat[self.target_name],data_cat[ftr_name]).values,
                                           correction=True)
        iv_val =iv_cal_cat(data_cat,self.target_name,var_name=ftr_name)
        iv_cat_got =iv_cat(iv_val)
        ############### Target Rate Dataframe ###################
        target_df=data_cat.groupby(ftr_name)[self.target_name].agg({'count','sum'}).reset_index(). \
            rename(columns={'count':'total','sum':'total_target'})
        
        target_df['rate']=round(target_df['total_target']/target_df['total'],2)
        
        #target_df.to_csv('target_df.csv')
        #data_cat.to_csv('data_cat.csv')

        #data_cat['hue']=data_cat[self.target_name].map(lambda x: str(x))
        data_cat['class']=data_cat[ftr_name].astype('category')
        
        
        ########################################
        # Create Table Data 
        col_label =("Measure", "Values")
        table_data =np.array([['IV Value','IV Category','Chi Sqr','Chi pval','Gini','Mutual Info'],
                    [round(iv_val,3),iv_cat_got,round(chi2,3),round(p,3),round(gini_val,3),round(mi_val,3)]]).T
        
        
        
        ########################################
        # Do Viz
        fig,ax=plt.subplots(1,2,figsize=(20,5))
        sns.countplot(x='class', hue=self.target_name, data=data_cat,ax=ax[0])
        ax[0].set_xlabel("Feature Names",fontsize=15)
        ax[0].xaxis.set_tick_params(rotation=45)
        ax[0].set_ylabel("Count",fontsize=15)
        ax_2 = ax[0].twinx()
        ax_2.plot(target_df[ftr_name].values,target_df['rate'].values,color='green',label='Target Rate',lw=2)
        ax_2.plot(target_df[ftr_name].values,[target_rate for _ in target_df[ftr_name].values],color='purple',label='Target Rate',lw=2)
               
        ax_2.set_ylabel("Target Rate",fontsize=15)
        ax_2.grid(False)
        
                
        ax[1].axis('off')
        data_table=ax[1].table(cellText=table_data,colLabels=col_label,loc='center',colWidths=[0.2]*2)
        data_table.auto_set_font_size(False)
        data_table.set_fontsize(15)
        data_table.scale(1, 3)
        fig.suptitle("Basic EDA for Feature={}".format(ftr_name),fontsize=15) 
        
    def conti_monobin_viz(self,ftr_name='age',bins=20):

        data_sum,iv_val,gini_val,mi_val,chi2, p=conti_mono_bin(self.data,self.target_name,var_name=ftr_name,n = bins)
        iv_cat_got =iv_cat(iv_val)

        sdata=self.data[[self.target_name,ftr_name]]

        target_rate =round(sdata[self.target_name].sum()/sdata.shape[0],3)
            
        # Create Table Data 
        col_label =("Measure", "Values")
        table_data =np.array([['IV Value','IV Category','Chi Sqr','Chi pval','Gini','Mutual Info'],
                [round(iv_val,3),iv_cat_got,round(chi2,3),round(p,3),round(gini_val,3),round(mi_val,3)]]).T

        # data prepration
        l=data_sum['min_'+str(ftr_name)].tolist()
        u=data_sum['max_'+str(ftr_name)].tolist()
        rate =data_sum['target_rate'].tolist()
        cat_gp =data_sum[ftr_name].tolist()

        
        sdata['grp_cat']=sdata.apply(lambda x:grp_manager(x,ftr_name,l,u,cat_gp),axis=1)
        sdata['class']=sdata['grp_cat'].astype('category')
        #data['target_class']=data[self.target_name].astype('category')
        sdata['target_class']=sdata[self.target_name].map(lambda x: '1' if x==1 else '0').astype('category')

        sdata.to_csv('data_cont.csv')
        #data_sum.to_csv('data_sum.csv')
        
        # Do Viz
        fig,ax=plt.subplots(1,2,figsize=(20,5))
        sns.countplot(x='class', hue='target_class', data=sdata,ax=ax[0],dodge=True)
        ax[0].set_xlabel("Feature Names",fontsize=15)
        ax[0].xaxis.set_tick_params(rotation=45)
        ax[0].set_ylabel("Count",fontsize=15)
        ax_2 = ax[0].twinx()
        ax_2.plot(cat_gp,rate,color='green',label='Target Rate',lw=2)
        ax_2.plot(cat_gp,[target_rate for _ in cat_gp],color='purple',label='Target Rate',lw=2)
               
        ax_2.set_ylabel("Target Rate",fontsize=15)
        ax_2.grid(False)
        
                
        ax[1].axis('off')
        data_table=ax[1].table(cellText=table_data,colLabels=col_label,loc='center',colWidths=[0.2]*2)
        data_table.auto_set_font_size(False)
        data_table.set_fontsize(15)
        data_table.scale(1, 3)
        fig.suptitle("Basic EDA for Feature={}".format(ftr_name),fontsize=15) 

            
    def kde_conti_dist(self,ftr_name='duration'):
        
        fig,ax = plt.subplots(1,1,figsize=(10,5))
        color =['orange','green','blue']
        color_dict ={}
        for k,v in enumerate(self.data[self.target_name].unique().tolist()):
            color_dict[v]=color[k]
            
        for item in self.data[self.target_name].unique().tolist():
            data_small=self.data[(self.data[self.target_name]==item)]
            ax = sns.kdeplot(data_small[ftr_name], shade=True, color=color_dict[item],label=str(item))
        ax.set_title("KDE Plots={}".format(ftr_name))
        ax.set_xlabel(ftr_name)
        ax.set_ylabel("Prob")

    def pair_conti_plot(self,ftr_list=['age','duration','campaign','pdays']):

        data_plot =self.data[ftr_list+[self.target_name]]
        g = sns.pairplot(data_plot, hue=self.target_name, palette="Set2", diag_kind="kde", height=2.5,
                 vars=ftr_list)
                
    def var_ranks_plot(self,output=True,msr_plot=None):

        num_iv_list=[]
        num_mi_list=[]

        cat_iv_list=[]
        cat_mi_list=[]

        for item in self.num_vars:
            
            data_sum,iv_val,gini_val,mi_val,chi2, p=conti_mono_bin(self.data,self.target_name,var_name=item,n = 20)
            num_mi_list.append(round(mi_val,3))
            num_iv_list.append(round(iv_val,3))

        for item in self.cat_vars:

            cat_iv_list.append(round(iv_cal_cat(self.data,self.target_name,var_name=item),3))
            cat_mi_list.append(round(mutual_info(self.data,item,self.target_name,munit='nats'),3))
            

        #print(num_iv_list)
        #print(num_mi_list)

        #print(cat_iv_list)
        #print(cat_mi_list)
        data_dict ={'feature_names':self.num_vars+self.cat_vars,'minfo':num_mi_list+cat_mi_list,'iv_val':num_iv_list+cat_iv_list}

        perf_data =pd.DataFrame(data_dict)
        perf_data['iv_cat']=perf_data['iv_val'].map(lambda x:iv_cat(x))

        if output:
             perf_data.to_csv(self.path+self.name+'_var_rank.csv')

        if msr_plot=='IV':
            perf_sorted =perf_data.sort_values(by=['iv_val'],ascending=False)
            fig, ax =plt.subplots(1,1,figsize=(10,5))
            ax.bar(perf_sorted['feature_names'],perf_sorted['iv_val'])
            
            ax.plot(perf_sorted['feature_names'],[0.03 for _ in perf_sorted['feature_names']],label='Medium',color='purple')
            ax.plot(perf_sorted['feature_names'],[0.5 for _ in perf_sorted['feature_names']],label='Strong',color='green')
            ax.xaxis.set_tick_params(rotation=45)        
            ax.set_title("IV Value Feature Rank",fontsize=15)              
            ax.legend(loc='best')
        
        elif msr_plot =='MI':
            perf_sorted =perf_data.sort_values(by=['minfo'],ascending=False)
            fig, ax =plt.subplots(1,1,figsize=(10,5))
            ax.bar(perf_sorted['feature_names'],perf_sorted['minfo'])                       
            ax.xaxis.set_tick_params(rotation=45)        
            ax.set_title("Mutual Info Feature Rank",fontsize=15)              
            #ax.legend(loc='best')

            
        
        

        

        

        
            

            
                
                

