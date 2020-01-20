#!/usr/bin/env python
# coding: utf-8

# # <center> Predicting Yield using Regression  </center>
# 
# 
# 
# | Feature | Description |
# | --- | --- |
# | farm_id | id given to the farms|
# | farm_area |  Area of the farm |
# | farming_company  | Name of the farming company |
# | deidentified_location  | location number assigned to the company |
# | date/timestamp | Timestamp value |
# | ingredient_type | ing_w, ing_x, ing_y, ing_z |
# | temp_obs  | observed temperature of that location |
# | cloudiness | cloud cover in that location |
# | wind_direction | Direction of wind of that location |
# | dew_temp | dew temperature of that location |
# | pressure_sea_level | Atmospheric pressure of the location |
# | wind_speed | Speed of the wind on that location |
# | num_processing_plants | Number of processing plant. |
# | operations_commencing_year  | year in which the company was started |
# | yield | the amount of prodcution of different igredient types|
# 
# 

# #**Exploratory Data Analysis¶**
# 

# **Mounting Google Drive** 
# 
# 

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# **Loading the libraries.**

# In[ ]:


import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt


# **Setting the Display Option right.**

# In[ ]:


def set_pandas_display_options() :

    pd.options.display.max_columns = 1000
    pd.options.display.max_rows = 1000
    pd.options.display.max_colwidth = 199
    pd.options.display.precision = 3

set_pandas_display_options()


# **Loading the Datasets.**

# In[ ]:


farm = pd.read_csv('/content/drive/My Drive/farm_data.csv')


# In[ ]:


weather = pd.read_csv('/content/drive/My Drive/train_weather.csv')


# In[ ]:


train = pd.read_csv('/content/drive/My Drive/train_data.csv')


# **Checking the head & shape of the farm dataset.**

# In[ ]:


farm.shape


# In[ ]:


farm.head()


# *I found that there are 1449 rows and 4 columns.* 
# 

# **Checking the Null Values**

# In[ ]:


percent_missing = farm.isnull().sum() * 100 / len(farm)
missing_value_farm = pd.DataFrame({'column_name': farm.columns,
                                 'percent_missing': percent_missing})
missing_value_farm


# *num_processing_plants & operations_commencing_year have very high Null values so we can drop them for now.*

# In[ ]:


del farm['num_processing_plants']
del farm['operations_commencing_year']


# In[ ]:


farm.head()


# **Checking the datatypes of the variables.**

# In[ ]:


farm.dtypes


# **Using describe to get some informantion about the dataset.**

# In[ ]:


farm.describe(include='all')


# 
# 
# 1.   farm_id (fid_29387) is the most occured one.
# 2.   farm_area starts with 26.292 and ends on 81290, which is highly skewed we will check more during visualization.
# 3. Obey Farm is the top in farming_company with 549 frequency.
# 4. from Deidentified_location location 5290 tops with 274 counts.
# 
# 

# In[ ]:


weather.shape


# In[ ]:


weather.head()


# *In weather data we have 139773 rows and 9 columns.*

# **Checking the null values for weather data.**

# In[ ]:


percent_missing = weather.isnull().sum() * 100 / len(weather)
missing_value_weather = pd.DataFrame({'column_name': weather.columns,
                                 'percent_missing': percent_missing})
missing_value_weather


# *We have null values in cloudiness, wind_direction, pressure_sea_level and precipitation.
# We wont drop them, we can impute them or do fillna.*

# In[ ]:


weather = weather.fillna(method = 'ffill').fillna(method = 'bfill')


# **Chceking the datatypes.**

# In[ ]:


weather.dtypes


# *As it is a timpestamp date we can extract the date using DatetimeIndex.
# This feature will help us to merge the different datasets later on.* 

# In[ ]:


weather['date'] = pd.DatetimeIndex(weather['timestamp'])


# In[ ]:


del weather['timestamp']


# In[ ]:


weather.describe(include= 'all')


# *After doing describe on weather data we can just see how the different parameters of the weather are varied. We can observe that there no invalid values in data. All the parameters are spread nicely.*

# In[ ]:


train.head()


# In[ ]:


train.shape


# *Train data is the biggest amongst the three with more than 2 crore rows and 4 columns. This is also timestamp data, so we need extract the date part from it.*

# In[ ]:


train['date'] = pd.DatetimeIndex(train['date'])


# In[ ]:


train.dtypes


# In[ ]:


train.describe(include= 'all')


# 
# 
# 1. The target variable is yield which is very highly skewed.
# 2. we have 4 ingredient types.
# 
# 

# **Finding the missing values:**

# In[ ]:


percent_missing = train.isnull().sum() * 100 / len(train)
missing_value_train = pd.DataFrame({'column_name': train.columns,
                                 'percent_missing': percent_missing})
missing_value_train


# **Merging:**

# *First will merge farm data with train groupby data on farm_id.
# Than we merge the merged data with groupby of weather on location and date.*

# In[ ]:


abc = pd.merge(farm, train, on = 'farm_id')


# In[ ]:


final_train = pd.merge(abc, weather, on = ['deidentified_location', 'date'])


# In[ ]:


final_train = pd.read_csv('/content/drive/My Drive/CSV/train_phd.csv')


# In[6]:


final_train.tail()


# In[ ]:


final_train.shape


# *So our final train data has more than 2.05 crore rows and 14 columns.*

# In[ ]:


final_train.dtypes


# In[ ]:


#percent_missing = final_train.isnull().sum() * 100 / len(final_train)
#missing_value_df = pd.DataFrame({'column_name': final_train.columns,
                                 'percent_missing': percent_missing})
#missing_value_df


# *Checking the unique values in our final train data.*
# 
# 

# In[ ]:


#output_lambda = final_train.apply(lambda x: [x.nunique()])
#output_lambda


# In[ ]:


#final_train.to_csv('/content/drive/My Drive/CSV/train_phd.csv', index = False)


# # **Visualization**

# **1. Correlation:**

# In[ ]:


pd.options.display.float_format = '{:.2f}'.format


# In[ ]:


corr = final_train.corr()


# In[9]:


print (corr['yield'].sort_values(ascending=False)[:6], '\n')
print (corr['yield'].sort_values(ascending=False)[-5:])


# 1. We can see that farm_area is on the top with yield in terms of correlation.
# Which is very likely as more the area more will be yield.
# 2. 2nd and 3rd are cloudiness and wind_speed we will check with other plots too.
# 3. Percipitation is negtively corelated which means more the rainfall less yield. This looks strange as rainfall is major factor, there might be some other factors depending upon which crop it is. We can always ask the domain expert and get more insights on it.

# **2. Heatmap:**

# In[ ]:


matrix = np.triu(final_train.corr())


# In[11]:


sns.heatmap(corr, annot= True, fmt= '.1g', vmin=-1, vmax=1, center= 0, cmap= 'coolwarm', linewidths=3, linecolor='black', mask=matrix)


# heatmap provides informantion to understand in seconds which are their best performing contents and which are less performing ones.
# Same thing we noticed with heatmap too. I have used half heatmap here.

# In[12]:


corr_matrix=final_train.corr()
plt.subplots(figsize=(20,15))

sns.heatmap(corr_matrix, annot=True, annot_kws={"size": 10})


# 
# 
# *   dew_temp & temp_obs have very high correlation among them so we can drop one of them
# *   same for wind_direction and wind_speed we can drop one of them
# *   pressure_sea_level i think i can drop this column too.
# 
# 

# In[13]:


final_train.columns


# In[ ]:


cat_cols = [ 'farming_company', 'deidentified_location', 'ingredient_type' ]


# In[15]:


num_cols = final_train.columns[~final_train.columns.isin(cat_cols)]
num_cols = num_cols.drop('yield')
num_cols


# **3. Distribution Plot(Target):**

# In[16]:


sns.distplot(final_train['yield'], bins = 20, kde = False, 
             hist_kws = {"alpha": 1}).set(xlabel = 'Yield', ylabel = 'Count')


# In[17]:


final_train['yield'].value_counts()


# We can see that our target variable is highly left skewed with large number of zero values in it making it highly skewed.
# We can transform this by taking log and normalize it.

# **4.Distribution plot (numerical columns):**

# In[18]:


fig, ax = plt.subplots(4, 2, figsize = (20, 20))

for var, subplot in zip(num_cols, ax.flatten()):
    sns.distplot(final_train[var], bins = 20, kde = False, ax = subplot, hist_kws = {"alpha": 1})


# 
# 
# *  Right Skew
#     1. Dew temp. 
# *  Left Skew
#     1. Wind_Speed
#     2. Percipition
#     3. Farm_area
# *  Normal
#     1. Pressure_sea_level
#     2. temp_obs
# 
# - Note: We can make cloudiness a categorical variable over that we can do binning also.   
# 
# 

# **5. Scatter Plot**

# In[ ]:


#fig, ax = plt.subplots(4, 2, figsize = (20, 30))

#for var, subplot in zip(num_cols, ax.flatten()):
    #sns.regplot(x = var, y = 'yield', data = final_train, 
                ax = subplot).set(xlabel = var, ylabel = 'yield')


# *Let us visualise bivariate relationship between target feature and other numeric features using scatter plot.*
# 
# *All we can see is that there are many outliers in all the columns.
# No columns show perfect relationship with the target varibale.
# se need to build outlier robost model that can tackle such situations.*
# 
# 

# **6. Playing With Correlation:**

# In[20]:


final_train[['yield', 'farm_area']].corr()


# In[21]:


print('# of observations : ', final_train[final_train['yield']> 50].shape[0])

df1 = final_train[final_train['yield']> 50]

df1[['farm_area', 'yield']].corr()


# *I tired to take values higher than 50 for yield and than check the corelation with farm area but it didnt changed much infact it decreased a bit. so not tampering it for now .*

# In[22]:


print('# of observations : ', final_train[final_train['farm_area']> 100].shape[0])

df1 = final_train[final_train['farm_area']> 100]

df1[['farm_area', 'yield']].corr()


# *Same i tried with farm_area alos but nothing fruitfull happend here.*

# **7. Box Plots and Count Plots:**

# In[ ]:


import matplotlib.pyplot as plt


# In[25]:


fig, ax = plt.subplots(3, 2, figsize = (20, 40))


for var, [subplotA, subplotB] in zip(cat_cols, ax):
    
    sns.countplot(final_train[var], ax = subplotA)
    for label in subplotA.get_xticklabels() :
        label.set_rotation(90)
        
    sns.boxplot(x = var, y = 'yield', data = final_train, ax = subplotB)
    for label in subplotB.get_xticklabels() :
        label.set_rotation(90)


# 
# 
# *   Farming_company
#      1. Obey farms have the largest counts among all and also very high outliers.
#      2 .After that we have Wayne farms, Senderson Farms, Dele Food Company and Del Monte Foods.
#      3. Rest have very low farms.
#      4. Senderson farms too have many outliers.
# *   Deindetified_location
#      1. location_2532 is the highest in this and also have the largest outliers.
#      2. Followed by location(8241, 5489, 5410, 5290)
#      3. Location 565 also have high outliers.
# *   Ingredient_type
#      1. Ingredient w has the highest count followed by x, y and z.
#      2. Ingredient y  has the highest outliers followed by x, z and w.
# 

# **8.PairPlot**

# *All of this informantion have explained above as pair plot have scatterplot and histogram which shows the spread of the data.*

# **9. Barplot**

# In[26]:


sns.barplot(x="ingredient_type", y="yield", data=final_train)


# *Ingredient y shows highest relationship with the target. 
# But the count of w was more than y which is quite suprising.*

# **10 . Catplot**

# In[27]:


sns.catplot(x="yield",y="farming_company",data=final_train)


# *As we knew Obey Farms had the highest count so it shows the strongest relationship with yield(target).*

# In[28]:


sns.catplot(x="yield", y="deidentified_location", data=final_train);


# *Same in Unindentified location, location_2532 was the highest so it has the maximum with the target variable.*

# In[ ]:


final_train['Month'] = pd.DatetimeIndex(final_train['date']).month 


# **11. Line Plot**

# In[32]:


sns.lineplot(x="date", y="yield", data=final_train)


# *When we check the yield for the enitre year (2016) we can see the yielding pattern*
# 
# 
# *   From March to June has the maximum yield. 
# *   This can be the farming season in that particluar area.
# *   There is no yield in the month of December, i think christmas might be the reason.
# 

# In[33]:


sns.lineplot(x="Month", y="yield", data=final_train)


# *Here i check month wise relationship with the target.*
# 
# 
# *   We can say farming season is from January ti June
# *   Highest yeild comes in March to June.
# *   July, August, September and October has almost no yield.
# *   December has no yield to this can be termed as Seasonality.
# 
# 
