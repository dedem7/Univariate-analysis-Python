#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy import stats
pd.options.mode.chained_assignment = None 
from sklearn.preprocessing import LabelEncoder
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import scipy.stats
from sklearn.linear_model import LinearRegression
from scipy import stats


df1=pd.read_csv("in/tables/price_comparison.csv")
#link:
#https://connection.eu-central-1.keboola.com/admin/projects/418/table-preview/out.c-testing-bucket.price_comparison?context=%2Fadmin%2Fprojects%2F418%2Ftransformations%2Fbucket%2F241951908%2Ftransformation%2F251386369


# ### Data preparation

# The main goal of investigation is to conduct univariate analysis(i.e. statistical tests of significance, dependency analysis, regression analysis, etc.)
# The target (dependent) variable is the "price_with_vat_czk",therefore we will evaluate relationship of other variables to target variable, especially 'seller_country' and 'model_family' since it was given as a task.

# In[3]:


#picking necessary columns for dataset df1
df1 = df1[['seller_country','model_family','power','fuel_type','transmission','year','drive','mileage','equipment_position','days_on_stock','price_with_vat_czk']]


# In[4]:


df1.head(3)


# In[5]:


#Eleven columns are in general and more than 2 mln rows are selected
df1.shape


# In[35]:


#ploting distribution of "price_with_vat_czk"
import matplotlib
import warnings
warnings.simplefilter(action='ignore')


ax1=df1.plot(x="model_family", y="price_with_vat_czk")
ax1.get_yaxis().set_major_formatter(
   matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

plt.show()


# The graph shows distribution of prices for each "model_family".
# There is high variation of car prices between 0 and 20,000,000, the spikes are price outliers.
# In order to avoid issues related to further analysis and negative influence on evaluation of models, the dataset which will not contain outliers will be created ("no_outler").

# In[36]:


no_outlier=df1[(df1['price_with_vat_czk']<20000000)]
ax2=no_outlier.plot(x="model_family", y="price_with_vat_czk")
ax2.get_yaxis().set_major_formatter(
   matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

plt.show()


# Now we can check the frequency of "seller_country" for following datasets:
# gu1 is original dataset
# gu2 is dataset that contains only outliers
# gu3 is the dataset without outliers
# 

# In[8]:


gu1=df1['seller_country'].value_counts()
gu2=df1.loc[df1['price_with_vat_czk']>20000000, 'seller_country'].value_counts()
gu3=df1.loc[df1['price_with_vat_czk']<20000000, 'seller_country'].value_counts()


# In[9]:


for x in [gu1,gu2,gu3]: print(x)


# For visually comparing these frequencies, pie charts will be plotted:

# In[10]:


fig = plt.figure(figsize=(20,20))

ax1 = plt.subplot2grid((1,3),(0,0))
plt.pie(gu1, autopct='%1.1f%%',labels=gu1.index,
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax1.legend(loc=3, labels=gu1.index)
plt.title('All dataset: \n {}'.format(gu1.to_string()))


ax2 = plt.subplot2grid((1,3),(0,1))
plt.pie(gu2, autopct='%1.1f%%',labels=gu2.index,
        shadow=True, startangle=90)
ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax2.legend(loc=3, labels=gu2.index)
plt.title('Outliers distribution (> 20,000,000): \n {}'.format(gu2.to_string()))


ax3 = plt.subplot2grid((1,3),(0,2))
plt.pie(gu3, autopct='%1.1f%%',labels=gu3.index,
        shadow=True,colors=("darkcyan","khaki"),startangle=90)
ax3.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax3.legend(loc=3, labels=gu3.index)
plt.title('No outliers distribution (< 20,000,000): \n {}'.format(gu3.to_string()))



plt.subplots_adjust(left=0.1,wspace=0.1, hspace=1, right=None, top=0.4, bottom=0.1)


# Three pie charts above describe the proportion of Czech and German car ads, where DE is ~30 times larger than CZ. This may lead to the problem of biasing the sample (Berksonian bias)
# The third pie chart(green/yellow) shows 71555 observatons for CZ, therefore, to avoid biasing the sample the same sample size will be randomly generated for DE. (CZ and DE sample sizes will be equal) 

# In[11]:


#create NOT biased dataset with no outliers (len(CZ) = len(DE))
no_outlier[no_outlier['seller_country']=='DE'].sample(n=gu3[1], random_state=1)
work_dataset=pd.concat([no_outlier[no_outlier['seller_country']=='CZ'], no_outlier[no_outlier['seller_country']=='DE'].sample(n=gu3[1], random_state=1)]).sample(frac=1)
work_dataset=work_dataset[['seller_country','model_family','fuel_type','transmission','drive','year','power','mileage','equipment_position','days_on_stock','price_with_vat_czk']].dropna()


# In[12]:


#This dataset will be selected as main dataset for analysis.
work_dataset.shape


# ### Descriptive statistics

# In[13]:


#DE
work_dataset[(work_dataset['seller_country']=='DE')].describe().apply(lambda s: s.apply(lambda x: format((round(x,1)))))


# In[14]:


#CZ
work_dataset[(work_dataset['seller_country']=='CZ')].describe().apply(lambda s: s.apply(lambda x: format((round(x,1)))))


# In[15]:


#Medians by model_family
work_dataset.groupby('model_family').median().apply(lambda s: s.apply(lambda x: format(round(x,1))))


# Correlation matrix below gives information that "price_with_vat_czk" has positive correlation with "year" and "power", but negative with "mileage"(which makes sense:lower mileage, higher price of car), nevertheless that Pearson's rho has moderate or even low correlation level.

# In[16]:



work_dataset.corr().style.background_gradient(cmap='summer')

#https://matplotlib.org/stable/gallery/color/colormap_reference.html(color)


# ### Dependency analysis

# To check the price difference (variation) for CZ and DE, the "price_with_vat_czk" will be divided 
# (categorized) into three categories:"cheap", "medium" and "expensive". Then using these categories, contingency table with percentages will be created and visualized as barchart and boxplots. 

# In[17]:


#segment variable is created for categorized prices
segment=pd.qcut(work_dataset['price_with_vat_czk'],3, labels=["cheap", "medium", "expensive"])
cont_table=pd.crosstab([work_dataset.seller_country],pd.qcut(work_dataset['price_with_vat_czk'],3, labels=["cheap", "medium", "expensive"]),margins = False).apply(lambda r: round((r/r.sum())*100), axis=1)
#pd.crosstab([work_dataset.seller_country], work_dataset['model_family'],margins = False).apply(lambda r: r/r.sum(), axis=1)
print(cont_table)
cont_table.plot.bar(stacked=True)
plt.legend(title='Car segment:')
plt.show()


# In[18]:


bp=work_dataset.boxplot(column=['price_with_vat_czk'], showfliers=False,by='seller_country',figsize=(5,5))
bp.get_yaxis().set_major_formatter(
   matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.show()


# Numerically and graphically we can conclude that in general the prices in DE are higher than in CZ.
# For example boxplot above shows that the median of DE is higher than in CZ. 50% of CZ prices are in range (from ~ 150k to 500k), while half prices of DE are in range from 200k to 690k, which is wider and higher than in CZ.
# 
# According to these results we can say that two sample means and medians are different.
# But the question is, if they are statistically different?
# We will check it by running different statistical significance tests.

# ### Tests of significance (hypothesis testing)

# There are two types of test approaches for dependency analysis: Parametric and Non parametric.
# 
# The difference between parametric and non parametric tests is whether the samples come from populations with normal distribution or not.
# 
# If the sample comes from population with normal distribution, then parametric tests (e.g. ANOVA, student t-test) are used.
# 
# If the sample comes from population that has a different distribution than normal, then non parametric test (e.g. Mann-Whitney, Kruskall-Wallis) will be used
# Therefore firstly, we need to test our data for normality.

# For testing normality, we will use Jarque Bera test. This test can be applied to all numerical columns from dataset.
# 
# The main column of interest is "price_with_vat_czk" or it's samples grouped by seller country.
# 

# In[19]:


#Jarque_Bera test
print(stats.jarque_bera(work_dataset['price_with_vat_czk']),'\n',
stats.jarque_bera(work_dataset.loc[(work_dataset['seller_country']=='DE', 'price_with_vat_czk')]),'\n',
stats.jarque_bera(work_dataset.loc[(work_dataset['seller_country']=='CZ', 'price_with_vat_czk')]))


# The test hypothesis: 
# 
# H0:data is normally distributed
# 
# H1: non H0
# 
# The p value of each test for samples is less than 0.05, it means that in all cases(groupped, non grouped) we reject null hypothesis.
# 
# Therefore further tests will be non parametric.

# Chi square independence test (https://www.statisticssolutions.com/non-parametric-analysis-chi-square/).
# 
# We check if two categories have assosiation between each other:
#     
# "seller_country" and "segment"
#     
# "model_family" and "seller_country"
# 
# "model_family" and "segment"

# In[1041]:


#Chisquare independence test (Pearson)
#This Chisquare test can be applied on contingency table
from scipy.stats import chi2_contingency
print(
scipy.stats.chi2_contingency(pd.crosstab(work_dataset['seller_country'],segment))[1],
scipy.stats.chi2_contingency(pd.crosstab(work_dataset['model_family'],work_dataset['seller_country']))[1],
scipy.stats.chi2_contingency(pd.crosstab(work_dataset['model_family'],segment))[1])


# H0:independence of categorical variables(columns)
# 
# H1:dependence (non H0)
# 
# p values in all cases are less than 0.05, we reject null hypothesis

# Since "segment" was initially continuous variable(price_with_vat_czk), during categorizing it, it might loose informative power
# (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1458573/)
# 
# Therefore we can use Spearman rank test: (https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient):

# In[1042]:


print(
stats.spearmanr(work_dataset['seller_country'],work_dataset['price_with_vat_czk'],axis=0),'\n',
stats.spearmanr(work_dataset['model_family'],work_dataset['price_with_vat_czk'],axis=0))
#Here we evaluate assosiation between seller_country and price_with_vat_czk


# Both two tests above gives evidence that dependency between tested variables(i.e. "price_with_vat_czk" vs "seller country", "model_family" vs "price_with_vat_czk") are statistically significant, because p value is less than 0.

# Mean comparisons. Test, if the difference of group means or medians are statistically significant.
# 
# Kruskal Wallis test is non parametric alternative of ANOVA.

# In[1043]:


stats.kruskal(work_dataset.loc[(work_dataset['seller_country']=='DE', 'price_with_vat_czk')],work_dataset.loc[(work_dataset['seller_country']=='CZ', 'price_with_vat_czk')])


# In[1044]:


stats.kruskal(work_dataset.loc[(work_dataset['year']==2015, 'price_with_vat_czk')],work_dataset.loc[(work_dataset['year']==2017, 'price_with_vat_czk')])


# In[1045]:


stats.kruskal(work_dataset.loc[(work_dataset['transmission']=='TRANSMISSION_AUTOMATIC', 'price_with_vat_czk')],work_dataset.loc[(work_dataset['transmission']=='TRANSMISSION_SEMI_AUTOMATIC', 'price_with_vat_czk')])[1]<0.01


# The difference between group medians are statistically significant(p<0.05)

# Also we can check if variances of groups are different. For this purpose, we can use Levene's test of homogeneity.
# 
# H0: two samples have equal variances
# 
# H1: non H0

# In[1046]:


#Levenes test homogeneity
scipy.stats.levene(work_dataset.loc[(work_dataset['seller_country']=='DE', 'price_with_vat_czk')],work_dataset.loc[(work_dataset['seller_country']=='CZ', 'price_with_vat_czk')])[1]<0.01


# Two samples have not equal variances.

# The graphs and tests give evidence and statistical significance that prices depend on categories such as seller country,model_family,transmission, year and other variables. 

# ### Regression analysis

# In this section we will build a regression model for evaluation the relatiopnship between "price_with_vat_czk" (dependent variable) and all variables that are available in dataset.
# 
# First we check for multicollinearity between variables. Multicollinearity is one of regression assumptions, because it can wrongly influence on model estimation and it's quality(e.g.R-squared)

# In[1047]:


#Multicollinearity check
def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)

#multicollinearity
calc_vif(work_dataset[['year','power','mileage','equipment_position','days_on_stock','price_with_vat_czk']])


# VIF(variance inflation factor) has high score for "year", meaning that "year" highly correlates with some another variable. Therefore we exclude "year" from regression model.

# Since the dataset besides continuous variable also contains catecorical, these categories has to be encoded as dummy variables(binary:1 and 0) by using OneHotEncoding algorithm:

# In[20]:


#one hot encoding to encode categorical variables as 1 and 0
work_dataset_encoded=pd.get_dummies(work_dataset, drop_first=True,columns = work_dataset.columns[:5])
work_dataset_encoded


# After decoding, the number of columns increased to 1153.

# In[ ]:


#Setting endogeneous and exogeneous variables for regression


# In[85]:


endog=work_dataset_encoded['price_with_vat_czk']
exog=work_dataset_encoded.drop(['price_with_vat_czk','year'],axis=1)


# In[86]:


#Regression model
from sklearn.linear_model import LinearRegression

regressor_OLS=sm.OLS(endog, exog = sm.add_constant(exog)).fit()
regressor_OLS.summary()


# The quality of model, based on it's determination coefficient(R squared) is 0.81, which indicates of good quality. The model itself is significant since probability of F statistic is less than 0.05, however some variables(mainly dummy variables) are not significant in this model, because some slopes have p value higher than 0.05.

# In[110]:


#number of not significant variables
len(regressor_OLS.pvalues[regressor_OLS.pvalues>0.05])


# In[111]:


#number of significant variables
#pd.options.display.max_rows = 10
len(regressor_OLS.pvalues[regressor_OLS.pvalues<0.05])


# In order to check whether the fitted model will have a good power for prediction, we will split data into train and test dataset, and then we will check for accuracy.
# 
# Since we are already aware that some regressors are not significant in model, such regressors will be excluded from the prediction model and those regressors which have p value less than 0.01 will remain in the model.
# 
# P.S. Regressors with p<0.01 were choosed for better performance of model, nevertheless p<0.05 is also considered to be significant)  
# 
# Lasso regression will be employed for prediction model.

# In[87]:


#Setting exog and enog
endog=np.log(work_dataset_encoded['price_with_vat_czk'])
exog=work_dataset_encoded[regressor_OLS.pvalues[regressor_OLS.pvalues<0.01].index[1:]]


# In[112]:


len(exog.columns)
#296 columns in exog


# In[89]:


#Split train and test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(exog,endog,test_size=0.2)


# In[90]:


Y_train.shape, Y_test.shape


# In[113]:


#Lasso regression
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

clf = linear_model.Lasso(alpha=0.1,fit_intercept=True)
clf.fit(X_train, Y_train)
Y_pred=clf.predict(X_test)


# In[114]:


#R squared
round(r2_score(Y_test, Y_pred),2)


# In[115]:


#RMSE
round(mean_squared_error(Y_test, Y_pred),2)


# Prediction model shows higher than average perfomance(R squared is 0.6), however MSE is 0.29, which is relatively good (small) enough

# ### Prediction interval (confidence interval)

# In[95]:



#get standard deviation of y_test
sum_errs = np.sum((Y_test - Y_pred)**2)
stdev = np.sqrt(1 / (len(Y_test) - 2) * sum_errs)
    
    
#get interval from standard deviation
one_minus_pi = 1 - 0.95
ppf_lookup = 1 - (one_minus_pi / 2)
z_score = stats.norm.ppf(ppf_lookup)
interval = z_score * stdev
#generate prediction interval lower and upper bound cs_24

lower, upper = Y_pred - interval, Y_pred + interval
    


# Plot confidence interval of linear regression  - 95% of confidence

# In[96]:


xs = range(1,len(Y_pred)+1)

f, ax = plt.subplots()
ax.plot(xs[:], Y_test, color='pink', marker='o', label='true')
ax.plot(xs[:], lower, color='limegreen', marker='o', label='lower', lw=0.5, markersize=2)
ax.plot(xs[:], Y_pred, color='aqua', marker='o', label='pred', markersize=2)
ax.plot(xs[:], upper, color='dodgerblue', marker='*', label='upper', lw=0.5, markersize=2)

ax.get_yaxis().set_major_formatter(
   matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

plt.legend()

plt.show()


# In[109]:


#Confidence intervals in numbers
print("lower","predicted","upper","true")
for i in range(10):
    print(round(np.exp(lower[i])),round(np.exp(Y_pred[i])),round(np.exp(upper[i])), round(np.exp(Y_test.values[i])))
   
               


# ### Conclusion

# According to univariate approach, the price depends on each category and feature of dataset, especially on those whose slope coefficients are high. Regression summary shows that the price does not depend on some car models, but depend on their other characteristics.
# 
# The predictive power of model based on this categories and features is sufficient, because 95% of confidence it contains a true values. Also R squared and MSE are intermediate.
# 
# Finally, this model can be improved and better investigated by following regression assumptions,transformations, standardisation, intersections and subjective estimation. 
# 
