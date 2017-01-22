#ROYAL RUMBLE I!!!
#ARENA: Loan applications dataset
#Wrestlers: SVC vs Naive Bayes vs Gradient Boosting vs Decision Tree vs Random Forest 

#PART 1: preparing the arena (WARNING, PART 1 IS LONG,SO YOU BETTER SKIP THIS PART AND READ DIRECTLY PART 2 TO CHECK THE WINNER OF ROYAL RUMBLE I)

#import pandas
import pandas as pd
#import numpy
import numpy as np

#read data
df = pd.read_csv(r"C:\Users\Alcatraz\Documents\LoanStats3a.csv", low_memory=False, skiprows=0, header=1)
# I replaced dtype=str by low_memory=False to keep types as done in R

df.dtypes

#have a look

df.head()

#check shape
df.shape


#Now I find how many NA's are for each column
df.isnull().sum(axis=0)

#now I will drop those columns having at least 80% of NaN values
filt = df.isnull().sum()/len(df) < 0.8
df = df.loc[:, filt]

#check NaN's again

df.isnull().sum()

#check shape
df.shape

#get rid of columns that I consider not important for random forest analysis

df=df.drop(labels=['desc','url','zip_code','addr_state','purpose','member_id','title'],axis=1)

df.shape        

#Now the objective is create a nuew column based on labels that I consider bad indicators.

#first check the distinct values of the colum 'status'
       
df['loan_status'].unique()

##>>> df['loan_status'].unique()
##array(['Fully Paid', 'Charged Off', 'Current', 'In Grace Period',
##       'Late (16-30 days)', 'Late (31-120 days)', 'Default', nan,
##       'Does not meet the credit policy. Status:Fully Paid',
##       'Does not meet the credit policy. Status:Charged Off'], dtype=object)


#Now I create a list to store the labels (which belong to the status column) indicating bad status. Here, 1 means bad and 0 means good
is_bad = []

for i in df['loan_status']: 
    is_bad.append(1 if i in ['Late (16-30 days)', 'Late (31-130 days)', 'Default', 'Charged Off','Does not meet the credit policy. Status:Charged Off'] else 0)



#now I will add the list 'is_bad' to my dataframe df

df['is_bad'] = is_bad
df['is_bad'].head()


# The column 'revol_util' should be numeric but it has a percentage symbol
df['revol_util'].head()

##>>> df['revol_util'].head()
##0    83.7%
##1     9.4%
##2    98.5%
##3      21%
##4    53.9%

#so I remove the symbol %. Use rstrip

df['revol_util']=df['revol_util'].str.rstrip('%')

##>>> df['revol_util'].head()
##0    83.7
##1     9.4
##2    98.5
##3      21
##4    53.9

#now convert 'revol_util' to numeric

df['revol_util']=pd.to_numeric(df['revol_util'])

#same procedure with 'int_rate'
df['int_rate'].head()
df['int_rate']=df['int_rate'].str.rstrip('%')
df['int_rate']=pd.to_numeric(df['int_rate'])

#now I check the types of df again to see the type of 'issue_d'. If it's
df.dtypes
df['issue_d'].head()
##>>> df['issue_d'].head()
##0    Dec-2011
##1    Dec-2011
##2    Dec-2011
##3    Dec-2011
##4    Dec-2011
df.dtypes

##issue_d                        object

#not date, I have to convert it.





df['issue_d']=pd.to_datetime(df['issue_d'],format="%b-%Y")        

#repeat the steps with 'earliest_cr_line','last_pymnt_d' and 'last_credit_pull_d'

df['earliest_cr_line']=pd.to_datetime(df['earliest_cr_line'],format="%b-%Y")  
df['last_pymnt_d']=pd.to_datetime(df['last_pymnt_d'],format="%b-%Y")  
df['last_credit_pull_d']=pd.to_datetime(df['last_credit_pull_d'],format="%b-%Y")  

#the columns 'application_type','term' and 'initial_list_status' seems not important
df['application_type'].unique()
##>>> df['application_type'].unique()
##array(['INDIVIDUAL', nan], dtype=object)

df['term'].unique()
##>>> df['term'].unique()
##array([' 36 months', ' 60 months', nan], dtype=object)

df['initial_list_status'].unique()
##>>> df['initial_list_status'].unique()
##array(['f', nan], dtype=object)

#Also, the column 'id' is irrelevant, so delete all of them
df=df.drop(labels=['id','emp_title','application_type','term','initial_list_status','loan_status'],axis=1)


#shape
df.shape
df.dtypes

#check how many columns are numeric
len(list(df.select_dtypes(include=[np.number]).columns.values))

#get numeric columns only
numeric_cols = df._get_numeric_data()
#have a look
numeric_cols.head()
numeric_cols.shape
numeric_cols.dtypes

#define a dataset containing the nonnumeric columns
remaining = []
for i in df.columns.values:
    if i not in df.select_dtypes(include=[np.number]).columns.values:
        remaining.append(i)

nonnumeric_cols=df[remaining]
nonnumeric_cols.head()
nonnumeric_cols.dtypes
nonnumeric_cols.isnull().sum()

#After having a look at those nonnumeric columns, I decided that the following columns are probably not important:
#'grade','sub_grade','emp_length','verification_status','pymnt_plan'. I will delete them

df=df.drop(labels=['grade','sub_grade','emp_length','verification_status','pymnt_plan',],axis=1)

#Now I should decide about those date type columns

#what's their length?
len(nonnumeric_cols['issue_d'].unique())
##>>> len(nonnumeric_cols['issue_d'].unique())
##56
len(nonnumeric_cols['earliest_cr_line'].unique())
##>>> len(nonnumeric_cols['earliest_cr_line'].unique())
##531
len(nonnumeric_cols['last_credit_pull_d'].unique())
##>>> len(nonnumeric_cols['last_credit_pull_d'].unique())
##111
len(nonnumeric_cols['last_pymnt_d'].unique())
##>>> len(nonnumeric_cols['last_pymnt_d'].unique())
##106

#Too many! Random forest will not work with so many distinct values for each variable, so I delete them
df=df.drop(labels=['issue_d','earliest_cr_line','last_credit_pull_d','last_pymnt_d'],axis=1)

#have a look
df.head()


#go back to numeric columns
#I will plot them, so I remove NaN's first

numeric_cols.isnull().sum()
numeric_cols=numeric_cols.dropna(axis=0)
numeric_cols.head()

#Now, the following is difficult to explain for me. I will plot each column against the column 'is_bad'. I have to do this manually because I'm getting several error messages when I use loops
#In each plot I will see two curves. If both curves look similar, then I discard this column from my dataset df. If they don't look similar, then I should analyze the column and decide whether or not I include it in my model

#For example, consider the first numeric column, indicated by 0 as follows
n=numeric_cols.ix[:,0]
#attach the column 'is_bad'
n['is_bad']=numeric_cols['is_bad']
#little warning, just ignore it:

##>>> n['is_bad']=numeric_cols['is_bad']
##__main__:1: SettingWithCopyWarning: 
##A value is trying to be set on a copy of a slice from a DataFrame
##See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy

#group by and plot
n.groupby('is_bad').plot(kind='density',title=numeric_cols.columns.values[0])



#the two curves look quite similar, so I will delete this column from my dataset df 

#I repeat the same procedure for columns i = 1,2, ..., 32

#From the plots, I discard those plots where the curves are similar. Next, I look at the remaining features and select some of them (those which I find relevant for my analysis)

#I decided to choose the following columns from the plots: 'delinq_2yrs','inq_last_6mths','pub_rec', and 'pub_rec_bankruptcies' 
#So i remove the remaining columns from the plots
selection =['delinq_2yrs','inq_last_6mths','pub_rec', 'pub_rec_bankruptcies']
to_be_deleted = []
#how does numeric_cols look like?
numeric_cols.shape
##>>> numeric_cols.shape
##(14262, 33)

#choose the columns to be deleted
for i in range(0,32):
    x=numeric_cols.columns.values[i] 
    if x in selection:
        pass
    else:
        to_be_deleted.append(x)

to_be_deleted
##>>> to_be_deleted
##['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'int_rate', 'installment', 'annual_inc', 'dti', 'mths_since_last_delinq', 'open_acc', 'revol_bal', 'revol_util', 'total_acc', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_amnt', 'collections_12_mths_ex_med', 'policy_code', 'acc_now_delinq', 'chargeoff_within_12_mths', 'delinq_amnt', 'tax_liens']

#delete them from df
df=df.drop(labels=to_be_deleted,axis=1)

#have a look
df.head()
##>>> df.head()
##  home_ownership  delinq_2yrs  inq_last_6mths  pub_rec  pub_rec_bankruptcies  \
##0           RENT          0.0             1.0      0.0                   0.0   
##1           RENT          0.0             5.0      0.0                   0.0   
##2           RENT          0.0             2.0      0.0                   0.0   
##3           RENT          0.0             1.0      0.0                   0.0   
##4           RENT          0.0             0.0      0.0                   0.0   
##
##   is_bad  
##0       0  
##1       1  
##2       0  
##3       0  
##4       0  


#IMPORTANT PARENTHESIS: when I was plotting I had problems with the following columns, i=22,23,25,26,27,28,29,31 yield a singular matrix error. For example, the column 22 yields the error:

##>>> n.groupby('is_bad').plot(kind='density',title=numeric_cols.columns.values[22])
##Traceback (most recent call last):
##  File "<stdin>", line 1, in <module>
##  File "C:\Users\Alcatraz\Anaconda2\lib\site-packages\pandas\core\groupby.py", line 311, in __call__
##    return self._groupby.apply(f)
##  File "C:\Users\Alcatraz\Anaconda2\lib\site-packages\pandas\core\groupby.py", line 651, in apply
##    return self._python_apply_general(f)
##  File "C:\Users\Alcatraz\Anaconda2\lib\site-packages\pandas\core\groupby.py", line 655, in _python_apply_general
##    self.axis)
##  File "C:\Users\Alcatraz\Anaconda2\lib\site-packages\pandas\core\groupby.py", line 1527, in apply
##    res = f(group)
##  File "C:\Users\Alcatraz\Anaconda2\lib\site-packages\pandas\core\groupby.py", line 647, in f
##    return func(g, *args, **kwargs)
##  File "C:\Users\Alcatraz\Anaconda2\lib\site-packages\pandas\core\groupby.py", line 309, in f
##    return self.plot(*args, **kwargs)
##  File "C:\Users\Alcatraz\Anaconda2\lib\site-packages\pandas\tools\plotting.py", line 3566, in __call__
##    **kwds)
##  File "C:\Users\Alcatraz\Anaconda2\lib\site-packages\pandas\tools\plotting.py", line 2645, in plot_series
##    **kwds)
##  File "C:\Users\Alcatraz\Anaconda2\lib\site-packages\pandas\tools\plotting.py", line 2441, in _plot
##    plot_obj.generate()
##  File "C:\Users\Alcatraz\Anaconda2\lib\site-packages\pandas\tools\plotting.py", line 1028, in generate
##    self._make_plot()
##  File "C:\Users\Alcatraz\Anaconda2\lib\site-packages\pandas\tools\plotting.py", line 2073, in _make_plot
##    stacking_id=stacking_id, **kwds)
##  File "C:\Users\Alcatraz\Anaconda2\lib\site-packages\pandas\tools\plotting.py", line 2127, in _plot
##    gkde = gaussian_kde(y, bw_method=bw_method)
##  File "C:\Users\Alcatraz\Anaconda2\lib\site-packages\scipy\stats\kde.py", line 171, in __init__
##    self.set_bandwidth(bw_method=bw_method)
##  File "C:\Users\Alcatraz\Anaconda2\lib\site-packages\scipy\stats\kde.py", line 488, in set_bandwidth
##    self._compute_covariance()
##  File "C:\Users\Alcatraz\Anaconda2\lib\site-packages\scipy\stats\kde.py", line 499, in _compute_covariance
##    self._data_inv_cov = linalg.inv(self._data_covariance)
##  File "C:\Users\Alcatraz\Anaconda2\lib\site-packages\scipy\linalg\basic.py", line 677, in inv
##    raise LinAlgError("singular matrix")
##numpy.linalg.linalg.LinAlgError: singular matrix

#Therefore, I decided to select manually those columns where I was having problems to plot and which could possibly be relevant for random forest 

#for example, I had a look at those columns where I was having problems

numeric_cols.ix[:,[22,23,25,26,27,28,29,31]].head()
##>>> numeric_cols.ix[:,[22,23,25,26,27,28,29,31]].head()
##    recoveries  collection_recovery_fee  collections_12_mths_ex_med  \
##3         0.00                      0.0                         0.0   
##4         0.00                      0.0                         0.0   
##16        0.00                      0.0                         0.0   
##18        0.00                      0.0                         0.0   
##27      260.96                      2.3                         0.0   
##
##    policy_code  acc_now_delinq  chargeoff_within_12_mths  delinq_amnt  \
##3           1.0             0.0                       0.0          0.0   
##4           1.0             0.0                       0.0          0.0   
##16          1.0             0.0                       0.0          0.0   
##18          1.0             0.0                       0.0          0.0   
##27          1.0             0.0                       0.0          0.0   
##
##    tax_liens  
##3         0.0  
##4         0.0  
##16        0.0  
##18        0.0  
##27        0.0  

#At the end, I decided not to choose any of these columns :), so i deleted them from df. They were already included in the list to_be_deleted

#END OF PARENTHESIS

#now check types
df.dtypes
#the columns 'home_ownership' and 'is_bad' are categorical, so use LabelEncoder

#first delete rows having NaN's
df.isnull().sum()
df=df.dropna(axis=0)
df=df.reset_index(drop=True)


#Now use LabelEncoder
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

df['home_ownership']=le.fit_transform(df['home_ownership'])
df['is_bad']=le.fit_transform(df['is_bad'])
#types
df.dtypes

#copy the column 'is_bad' and drop it
y=pd.Series(df['is_bad'].copy())

df=df.drop(labels=['is_bad'],axis=1)

#define training and testing sets
from sklearn.cross_validation import train_test_split
df_train,df_test,y_train,y_test=train_test_split(df,y,test_size=0.2)

df_test.shape
##>>> df_test.shape
##(8234, 5)
df_train.shape
##>>> df_train.shape
##(32936, 5)



#PART 2: FINALLY ROYAL RUMBLE!!!

#random forest!
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()

rf.fit(df_train,y_train)

predictions1 = rf.predict(df_test)

from sklearn.metrics import accuracy_score

accuracy_score(predictions1,y_test,normalize=True)
##>>> accuracy_score(predictions1,y_test,normalize=True)
##0.85669176584891915


#another way to measure accuracy
rf.score(df_test,y_test)

#The score of random forest is 0.85669176584891915

#decision tree!
from sklearn import tree

dt = tree.DecisionTreeClassifier()

dt.fit(df_train,y_train)

predictions2 = dt.predict(df_test)

accuracy_score(predictions2,y_test,normalize=True)
##>>> accuracy_score(predictions2,y_test,normalize=True)
##0.85608452756861797

#The score of decision tree is 0.85608452756861797


#Gradient boosting!
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier()

gb.fit(df_train,y_train)

predictions3=gb.predict(df_test)
accuracy_score(predictions3,y_test,normalize=True)
##>>> accuracy_score(predictions3,y_test,normalize=True)
##0.85729900412922033

#The score of gradient boosting is 0.85729900412922033

#Naive Bayes!
from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(df_train,y_train)
predictions4=gnb.predict(df_test)
accuracy_score(predictions4,y_test,normalize=True)
##>>> accuracy_score(predictions4,y_test,normalize=True)
##0.81467087685207673

#The score of naive Bayes is 0.81467087685207673

#SVC!
from sklearn.svm import SVC

svc=SVC()

svc.fit(df_train,y_train)
predictions5=svc.predict(df_test)
accuracy_score(predictions5,y_test,normalize=True)
##>>> accuracy_score(predictions5,y_test,normalize=True)
##0.85729900412922033

#The score of SVC is 0.85729900412922033

#There is a draw between SVC and Gradient Boosting!

#let's change the training and testing datasets

df_train,df_test,y_train,y_test=train_test_split(df,y,test_size=0.5)

svc.fit(df_train,y_train)
final_prediction1 = svc.predict(df_test)
accuracy_score(final_prediction1,y_test,normalize=True)
##>>> accuracy_score(final_prediction1,y_test,normalize=True)
##0.85343696866650476

#The score of SVC is 0.85343696866650476

gb.fit(df_train,y_train)
final_prediction2=gb.predict(df_test)
accuracy_score(final_prediction2,y_test,normalize=True)
##>>> accuracy_score(final_prediction2,y_test,normalize=True)
##0.85329123147923247

#The score of Gradient Boosting is 0.85329123147923247

##THE WINNER OF ROYAL RUMBLE I IS SVC!!!!!!!

#CONGRATULATIONS TO THE CHAMPION: SVC
