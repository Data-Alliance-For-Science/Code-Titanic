#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import matplotlib.pyplot as plt

# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


dftrain = pd.read_csv("train.csv")
dftest = pd.read_csv("test.csv")
dfgendersubmission = pd.read_csv("gender_submission.csv")
combine = [dftrain, dftest]


# # Data Dictionary #
survival:       Survival 0 = No, 1 = Yes pclass Ticket class 1 = 1st, 2 = 2nd, 3 = 3rd sex Sex Age Age in years
sibsp:          # of siblings / spouses aboard the Titanic
parch:          # of parents / children aboard the Titanic
ticket:         Ticket number
fare:           Passenger fare
cabin:          Cabin number
embarked:       Port of Embarkation C = Cherbourg, Q = Queenstown, S = Southampton Variable Notes

pclass:         A proxy for socio-economic status (SES) 1st = Upper 2nd = Middle 3rd = Lower

age:            Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

sibsp:          The dataset defines family relations in this way. Sibling = brother, sister, stepbrother, stepsister Spouse = husband, wife (mistresses and fiancés were ignored)

parch:          The dataset defines family relations in this way… Parent = mother, father Child = daughter, son, stepdaughter, stepson Some children travelled only with a nanny, therefore parch=0 for them 
# # Exploratory Analysis #  

# In[3]:


#Dimensión de la base de datos
dftrain.shape 


# In[4]:


#Información sobre las variables del dataframe
dftrain.info()


# In[5]:


#Database heading#
dftrain.head()


# In[6]:


# Dropping non useful variables
dftrain = dftrain.drop(["Name","Ticket","Fare", "Cabin"],axis=1) 
dftrain


# In[7]:


#Total missing values#
dftrain.isnull().sum()


# In[34]:


#Replacing NaN with 0's
dftrain = dftrain.fillna(0)
dftrain.isnull().sum()


# # 1.2 Embarked Class treatment #

# In[31]:


# Embarked - Unique values
matriz= dftrain['Embarked'].unique() 
len(matriz) 
print(matriz)


# In[28]:


#Most common Embarked class#
freq_port = dftrain.Embarked.dropna().mode()[0]
freq_port


# In[29]:


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
dftrain[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[30]:


#Converting Embarkedfill categorical feature to numeric#
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(float)

dftrain.head()


# ## 1.2 Age treatment ## 

# In[4]:


#Empty array to contain guessed Age values based on Pclass x Gender combinations#
guess_ages = np.zeros((2,3))
guess_ages


# In[5]:


#Iterate over Sex (0 or 1) and Pclass (1, 2, 3) to calculate guessed values of Age for the six combinations.

for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) &                                   (dataset['Pclass'] == j+1)]['Age'].dropna()
            age_guess = guess_df.median()
            guess_ages[i,j] = ( age_guess/0.5 + 0.5 ) * 0.5

        for i in range(0, 2):
            for j in range(0, 3):
                 dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(float)

dftrain.head()


# In[6]:


#Age bands and correlations with Survived
dftrain['AgeBand'] = pd.cut(dftrain['Age'], 5)
dftrain[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)


# In[18]:


#Replace Age with ordinals based on these bands#
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
dftrain.head()


# In[19]:


# grid = sns.FacetGrid(dftrain, col='Pclass', hue='Gender')
grid = sns.FacetGrid(dftrain, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()


# In[24]:


#Changing Index for Id Passanger 
dftrain = dftrain.set_index("PassengerId")
dftrain


# In[25]:


# Sex - Unique Values
matriz2= dftrain['Sex'].unique()
len(matriz2) 


# ## Demographic Analysis: ### 

# In[37]:


#Checking the total number of passagers:
dftrain['Sex'].value_counts() 


# In[36]:


# Percentage of women who survived:
women = dftrain.loc[dftrain.Sex == "female"]["Survived"]
rate_women = sum(women)/len(women)
print("% of women who survived:", rate_women)


# In[38]:


# Percentage of male who survived:
male = dftrain.loc[dftrain.Sex == "male"]["Survived"]
rate_male = sum(male)/len(male)
print("% of male who survived:", rate_male)


# In[39]:


#Total persons according to class
dftrain['Pclass'].value_counts() 


# In[40]:


#Percentage of survivors according to class
firstclass = dftrain.loc[dftrain.Pclass == 1]["Survived"]
rate_firstclass = sum(firstclass)/len(firstclass)
print("% of first class peoples who survived", rate_firstclass)

secondclass = dftrain.loc[dftrain.Pclass == 2]["Survived"]
rate_secondclass = sum(secondclass)/len(secondclass)
print("% of second class peoples who survived", rate_secondclass)

thirdclass = dftrain.loc[dftrain.Pclass == 3]["Survived"]
rate_thirdclass = sum(thirdclass)/len(thirdclass)
print("% of third class peoples who survived", rate_thirdclass)


# In[41]:


dftrain.describe()


# In[42]:


# Sex - Ordinal to numerical
dftrain['Sex'].replace(['female','male'],[0,1],inplace=True) 


# In[43]:


# Sex - Graph
dftrain.groupby('Sex').size().plot(kind='bar')
plt.title('Distribución por género train ')
plt.show()


# In[44]:


#Survival probabilit
porcent_sobrevivientes = (dftrain[dftrain.Survived
                             > 0]['Survived'].count() * 1.0
       / dftrain['Survived'].count()) * 100.0
print("El porcentaje de sobrevivientes de la base de datos es {0:.2f}%"
      .format(porcent_sobrevivientes))


# # 2. Subset creation: Survivors #

# In[46]:


dftrain_sobrevivientes = dftrain[dftrain.Survived > 0]
dftrain_sobrevivientes                               


# In[48]:


#Total data#
len(dftrain_sobrevivientes) 


# In[49]:


# Graph - Sex survivors
dftrain_sobrevivientes.groupby('Sex').size().plot(kind='bar')
plt.title('Distribución por género')
plt.show()  #Male=1 Female=0


# In[50]:


# Age according to gender
dftrain_sobrevivientes[(dftrain_sobrevivientes.Age <= 100)
             & (dftrain_sobrevivientes.Sex.isin(['0', '1'])
               )][['Age', 'Sex']].boxplot(by='Sex')
plt.title('edad segun el género')
plt.show()


# In[51]:


# Survivor women's average age
dftrain_sobrevivientes[dftrain_sobrevivientes.Sex == 0][['Age']].mean()


# In[52]:


#Average class survivors
dftrain_sobrevivientes[['Pclass']].mean()


# In[53]:


# Survivor men's avarage age
dftrain_sobrevivientes[dftrain_sobrevivientes.Sex == 1][['Age']].mean()


# In[54]:


dftrain_sobrevivientes.groupby('Pclass').size().plot(kind='bar')
plt.title('Distribución por clase')
plt.show() 


# In[55]:


pclass_gender_survival_count_df= dftrain.groupby(['Pclass','Sex'])['Survived'].sum()
dftrain.groupby(['Pclass','Sex']).count()
dftrain['count'] = 1 # agregar columna
dftrain.groupby(['Pclass','Sex','count']).count()
dftrain.groupby(['Pclass']).sum()


# In[56]:


pclass_gender_survival_count_df


# In[7]:


pclass_gender_survival_count_df= dftrain.groupby(['Pclass','Age'])['Survived'].sum()
pclass_gender_survival_count_df


# In[ ]:




