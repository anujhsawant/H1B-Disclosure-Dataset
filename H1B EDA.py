#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)
from plotly import tools
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df = pd.read_csv('H-1B_Disclosure_Data_FY17.csv')


# In[3]:


df.head()


# In[4]:


df = df.iloc[:,2:]


# In[5]:


df.head()


# In[6]:


df.isnull().sum()['EMPLOYER_BUSINESS_DBA']


# In[7]:


drop_cols = []
for col in list(df.columns):
    if(df.isnull().sum()[col]>500000):
        drop_cols.append(col)
df.drop(drop_cols,1,inplace=True)


# In[8]:


df.isnull().sum()


# In[9]:


df.shape


# In[10]:


df.drop(['EMPLOYER_PHONE'],1,inplace=True)


# In[11]:


df.isnull().sum()


# In[12]:


df.AGENT_ATTORNEY_NAME.value_counts()


# In[13]:


df.head()


# In[14]:


df.dtypes


# In[15]:


drop_cols = ['EMPLOYER_ADDRESS','EMPLOYER_CITY','AGENT_ATTORNEY_NAME','AGENT_ATTORNEY_CITY','AGENT_ATTORNEY_STATE','SOC_NAME','WORKSITE_CITY','WORKSITE_COUNTY']
df.drop(drop_cols,1,inplace=True)


# In[16]:


df.SUPPORT_H1B.value_counts()


# In[17]:


df.VISA_CLASS.value_counts(0)


# In[18]:


df.CASE_STATUS.value_counts()


# In[19]:


cols = ['CONTINUED_EMPLOYMENT','CHANGE_PREVIOUS_EMPLOYMENT','FULL_TIME_POSITION','PW_UNIT_OF_PAY','WAGE_UNIT_OF_PAY','WILLFUL_VIOLATOR']
for col in cols:
    print(df[col].value_counts())


# In[20]:


df.isnull().sum()


# In[21]:


df.LABOR_CON_AGREE.value_counts()


# In[22]:


df.PW_WAGE_LEVEL.value_counts()


# In[23]:


df.AGENT_REPRESENTING_EMPLOYER.value_counts()


# In[24]:


df.loc[df.AGENT_REPRESENTING_EMPLOYER.isna(),'AGENT_REPRESENTING_EMPLOYER']='X'
df.loc[df.LABOR_CON_AGREE.isna(),'LABOR_CON_AGREE']='X'
df.loc[df.SUPPORT_H1B.isna(),'SUPPORT_H1B']='X'
df.loc[df.EMPLOYMENT_START_DATE.isna(),'EMPLOYMENT_START_DATE']=df.EMPLOYMENT_START_DATE.mode()[0]
df.loc[df.EMPLOYMENT_END_DATE.isna(),'EMPLOYMENT_END_DATE']= df.EMPLOYMENT_END_DATE.mode()[0]
df.loc[df.PW_SOURCE_OTHER.isna(),'PW_SOURCE_OTHER']=df.PW_SOURCE_OTHER.mode()[0]
df.loc[df.PW_WAGE_LEVEL.isna(),'PW_WAGE_LEVEL']='Level V'


# In[25]:


df.isna().sum()


# In[26]:


df.loc[df.EMPLOYER_POSTAL_CODE.isna(),'EMPLOYER_POSTAL_CODE']=df.EMPLOYER_POSTAL_CODE.mode()[0]
df.loc[df.EMPLOYER_NAME.isna(),'EMPLOYER_NAME']= df.EMPLOYER_NAME.mode()[0]


# In[27]:


df.isna().sum()


# In[28]:


df.CASE_STATUS.value_counts()


# In[29]:


df.loc[df.H1B_DEPENDENT.isna(),'H1B_DEPENDENT']=df.H1B_DEPENDENT.mode()[0]
df.loc[df.WILLFUL_VIOLATOR.isna(),'WILLFUL_VIOLATOR']= df.WILLFUL_VIOLATOR.mode()[0]
df.loc[df.EMPLOYER_COUNTRY.isna(),'EMPLOYER_COUNTRY']= df.EMPLOYER_COUNTRY.mode()[0]


# In[30]:


df.dropna(inplace=True)


# In[31]:


df.shape


# In[32]:


df['DECISION_DURATION'] = abs(pd.to_datetime(df.CASE_SUBMITTED) - pd.to_datetime(df.DECISION_DATE))
df['EMPLOYMENT_DURATION'] = abs(pd.to_datetime(df.EMPLOYMENT_START_DATE) - pd.to_datetime(df.EMPLOYMENT_END_DATE))


# In[33]:


df.drop(['CASE_SUBMITTED','DECISION_DATE','EMPLOYMENT_START_DATE','EMPLOYMENT_END_DATE'],axis=1,inplace=True)


# In[34]:


df.isna().sum()


# In[35]:


df.head()


# In[36]:


df['EMPLOYMENT_DURATION'] = df['EMPLOYMENT_DURATION'].dt.days
df['DECISION_DURATION'] = df['DECISION_DURATION'].dt.days


# In[37]:


df.head()


# In[38]:


df.to_csv('cleaned_h1b.csv',index=False)


# In[39]:


# Checking number of applications state wise.
df.EMPLOYER_STATE.value_counts(ascending=True)[-15:].plot(kind='barh')
plt.show()


# In[40]:


df[df.EMPLOYER_STATE=='CA'].groupby('CASE_STATUS').size().plot(kind='bar')
plt.title('H1B Approval Count IN CA')
plt.show()


# In[41]:


df.dtypes


# In[42]:


df['countvar'] = 1
dftop = df.groupby('EMPLOYER_NAME',as_index=False).count().sort_values('countvar',ascending= False)[['EMPLOYER_NAME','countvar']][0:30]
t1 = go.Bar(x=dftop.EMPLOYER_NAME.values,y=dftop.countvar.values,name='top30employer')
layout = go.Layout(dict(title= "TOP EMPLOYERS SPONSORING H1B APPLICATIONS",yaxis=dict(title="Number of applications")))
data = [t1]
fig =go.Figure(data,layout)
iplot(fig)


# In[43]:


dftop = df.groupby('EMPLOYER_STATE',as_index=False).count().sort_values('countvar',ascending= False)[['EMPLOYER_STATE','countvar']][0:30]
t1 = go.Bar(x=dftop.EMPLOYER_STATE.values,y=dftop.countvar.values,name='top30employerstate')
layout = go.Layout(dict(title= "TOP STATES BY NUMBER OF H1B APPLICATIONS",yaxis=dict(title="Number of applications")))
data = [t1]
fig =go.Figure(data,layout)
iplot(fig)


# In[44]:


dftop = df.groupby('CASE_STATUS',as_index=False).count().sort_values('countvar',ascending= False)[['CASE_STATUS','countvar']]
t1 = go.Bar(x=dftop.CASE_STATUS.values,y=dftop.countvar.values,name='top30')
layout = go.Layout(dict(title= "NUMBER OF APPLICATIONS BY CASE STATUS ",yaxis=dict(title="Number of applications")))
data = [t1]
fig =go.Figure(data,layout)
iplot(fig)


# In[45]:


# Number of applications State wise by case status.

# California - CA
dftop = df[df.EMPLOYER_STATE=='CA'].groupby('CASE_STATUS',as_index=False).count().sort_values('countvar',ascending= False)[['CASE_STATUS','countvar']]
t1 = go.Bar(x=dftop.CASE_STATUS.values,y=dftop.countvar.values,name='top30')
layout = go.Layout(dict(title= "NUMBER OF APPLICATIONS IN <b>CALIFORNIA</b> BY CASE STATUS ",yaxis=dict(title="Number of applications")))
data = [t1]
fig =go.Figure(data,layout)
ca = iplot(fig)

# Texas - TX
dftop = df[df.EMPLOYER_STATE=='TX'].groupby('CASE_STATUS',as_index=False).count().sort_values('countvar',ascending= False)[['CASE_STATUS','countvar']]
t1 = go.Bar(x=dftop.CASE_STATUS.values,y=dftop.countvar.values,name='top30')
layout = go.Layout(dict(title= "NUMBER OF APPLICATIONS IN <b>Texas</b> BY CASE STATUS ",yaxis=dict(title="Number of applications")))
data = [t1]
fig =go.Figure(data,layout)
tx = iplot(fig)

# New Jersey - NJ
dftop = df[df.EMPLOYER_STATE=='NJ'].groupby('CASE_STATUS',as_index=False).count().sort_values('countvar',ascending= False)[['CASE_STATUS','countvar']]
t1 = go.Bar(x=dftop.CASE_STATUS.values,y=dftop.countvar.values,name='top30')
layout = go.Layout(dict(title= "NUMBER OF APPLICATIONS IN <b>New Jersey</b> BY CASE STATUS ",yaxis=dict(title="Number of applications")))
data = [t1]
fig =go.Figure(data,layout)
nj = iplot(fig)

# New York - NY
dftop = df[df.EMPLOYER_STATE=='NY'].groupby('CASE_STATUS',as_index=False).count().sort_values('countvar',ascending= False)[['CASE_STATUS','countvar']]
t1 = go.Bar(x=dftop.CASE_STATUS.values,y=dftop.countvar.values,name='top30')
layout = go.Layout(dict(title= "NUMBER OF APPLICATIONS IN <b>New York</b> BY CASE STATUS ",yaxis=dict(title="Number of applications")))
data = [t1]
fig =go.Figure(data,layout)
ny = iplot(fig)


# In[46]:


# Univariate Analysis
numerical = df.select_dtypes(include=np.number)
categorical = df.select_dtypes(exclude=np.number)


# In[47]:


numerical.shape


# In[48]:


dftop0 = df[df.AGENT_REPRESENTING_EMPLOYER=='Y'].groupby('CASE_STATUS',as_index=False).count().sort_values('countvar',ascending= False)[['CASE_STATUS','countvar']]
dftop1 = df[df.AGENT_REPRESENTING_EMPLOYER=='N'].groupby('CASE_STATUS',as_index=False).count().sort_values('countvar',ascending= False)[['CASE_STATUS','countvar']]
t0 = go.Bar(x=dftop0.CASE_STATUS.values,y=dftop0.countvar.values,name='Employers with Agent')
t1 = go.Bar(x=dftop1.CASE_STATUS.values,y=dftop1.countvar.values,name='Employers without Agent')
layout = go.Layout(dict(title= "NUMBER OF APPLICATIONS WITH/WITHOUT <b>AGENT</b> BY CASE STATUS ",yaxis=dict(title="Number of applications")))
data = [t0,t1]
fig =go.Figure(data,layout)
ny = iplot(fig)


# In[49]:


df['EMPLOYER_NAME'].value_counts()


# In[50]:


dftop = df.groupby('EMPLOYER_NAME',as_index=False).count()
dftop = dftop.sort_values('countvar',ascending= False)[['EMPLOYER_NAME','countvar']][0:30]
dftop1 = df.groupby(['EMPLOYER_NAME','CASE_STATUS'],as_index=False).count()
dftop1=dftop1[dftop1.EMPLOYER_NAME.isin(dftop.EMPLOYER_NAME)]
t1 = go.Bar(x=dftop1[dftop1.CASE_STATUS == 'CERTIFIED'].sort_values('countvar',ascending= False)['EMPLOYER_NAME'].values,y=dftop1[dftop1.CASE_STATUS == 'CERTIFIED'].sort_values('countvar',ascending= False)['countvar'].values,name='CERTIFIED')
t2 = go.Bar(x=dftop1[dftop1.CASE_STATUS == 'CERTIFIED-WITHDRAWN'].sort_values('countvar',ascending= False)['EMPLOYER_NAME'].values,y=dftop1[dftop1.CASE_STATUS == 'CERTIFIED-WITHDRAWN'].sort_values('countvar',ascending= False)['countvar'].values,name='CERTIFIED-WITHDRAWN')
t3 = go.Bar(x=dftop1[dftop1.CASE_STATUS == 'DENIED'].sort_values('countvar',ascending= False)['EMPLOYER_NAME'].values,y=dftop1[dftop1.CASE_STATUS == 'DENIED'].sort_values('countvar',ascending= False)['countvar'].values,name='DENIED')
t4 = go.Bar(x=dftop1[dftop1.CASE_STATUS == 'WITHDRAWN'].sort_values('countvar',ascending= False)['EMPLOYER_NAME'].values,y=dftop1[dftop1.CASE_STATUS == 'WITHDRAWN'].sort_values('countvar',ascending= False)['countvar'].values,name='WITHDRAWN')

data = [t1,t2,t3,t4]
layout = go.Layout((dict(title= "CASE STATUS OF TOP EMPLOYERS SPONSORING H1B APPLICATIONS",yaxis=dict(title="Number of applications"))),barmode='stack'
)

fig =go.Figure(data,layout)
iplot(fig)


# In[51]:


df['countvar'] = 1
dftop = df.groupby('JOB_TITLE',as_index=False).count().sort_values('countvar',ascending= False)[['JOB_TITLE','countvar']][0:30]
t1 = go.Bar(x=dftop.JOB_TITLE.values,y=dftop.countvar.values,name='jobtitle')
layout = go.Layout(dict(title= "TOP JOB PROFILES FOR H-1B APPLICATIONS",yaxis=dict(title="Number of applications")))
data = [t1]
fig =go.Figure(data,layout)
iplot(fig)


# In[52]:


df[df.DECISION_DURATION<20].DECISION_DURATION.hist()
plt.xlabel('Decision Duration in days')
plt.ylabel('Number of applications')
plt.title('Decision Duration v/s Number of applications ')


# In[ ]:





# In[ ]:




