import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,accuracy_score,classification_report,roc_auc_score,roc_curve
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.pipeline import Pipeline
import shap
import joblib
pd.set_option("display.max_columns",None)
df=pd.read_csv("Telco-Customer-Churn.csv")
#print(df.shape)
#print(df.describe(include='all'))
#print(df.isnull().sum())
#print((len(df[df['Churn']=='Yes'])/len(df))*100)
#print(df['TotalCharges'].tail(40))
df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')
#print(df[df['TotalCharges'].isnull()])
df['TotalCharges']=df['TotalCharges'].replace(np.nan,0)
#print(df['TotalCharges'].describe())
#print(df['customerID'].duplicated().sum())
#print(df['Contract'].value_counts())
contract_churn=pd.crosstab(index=df["Churn"],columns=df["Contract"],normalize='columns')*100
#print(contract_churn.round(1))
tenure_churn=df.groupby('Churn')['tenure'].mean()
#print(tenure_churn)
monthycharges_churn=df.groupby('Churn')['MonthlyCharges'].mean()
#print(monthycharges_churn)
internetservice_churn=pd.crosstab(index=df['Churn'],columns=df['InternetService'],normalize='columns')*100
#print(internetservice_churn.round(1))
paymentmethod_churn=pd.crosstab(index=df['Churn'],columns=df['PaymentMethod'],normalize='columns')*100
#print(paymentmethod_churn.round(1))
seniorciti_churn=pd.crosstab(index=df['Churn'],columns=df['SeniorCitizen'],normalize='columns')*100
#print(seniorciti_churn.round(1))
contract_internetservice=pd.crosstab(index=df['Contract'],columns=df['InternetService'])
#print(contract_internetservice)
'''sns.countplot(data=df,x='Churn',hue='Churn')
plt.title("frequency of churn")
plt.savefig("number of churn")
plt.show()
sns.countplot(data=df,x='Contract',hue='Churn')
plt.title("Churn frequency by Contract")
plt.savefig("Churn frequency on Contract")
plt.show()
sns.boxplot(data=df,x='Churn',y='tenure',hue='Churn')
plt.title('Churn frequency by tenure')
plt.savefig("Churn frequency by tenure")
plt.show()'''
corr_matrix=df[['MonthlyCharges','TotalCharges','tenure']]
#print(corr_matrix.corr())
#print(df.groupby('Churn')['TotalCharges'].mean())
bin_div=np.linspace(df['tenure'].min(),df['tenure'].max(),5)
bin_names=['New','Early','Established','Long-term']
df['tenure-bins']=pd.cut(df['tenure'],bins=bin_div,labels=bin_names,include_lowest=True)
tenure_catogary_churn=pd.crosstab(index=df["Churn"],columns=df["tenure-bins"],normalize='columns')*100
#print(tenure_catogary_churn.round(1))
test_rel_gender_churn=pd.crosstab(index=df['gender'],columns=df['Churn'],normalize='index')
test_rel_PaperlessBilling_churn=pd.crosstab(index=df['PaperlessBilling'],columns=df['Churn'],normalize='index')
'''print(test_rel_gender_churn.round(1))
print(test_rel_PaperlessBilling_churn.round(1))'''
dum_churn=pd.get_dummies(df['Churn'],drop_first=True,dtype=int)
df=pd.concat([df,dum_churn],axis=1)
#print(df.describe(include='all'))
encode=['gender','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingMovies','StreamingTV','Contract','PaperlessBilling','PaymentMethod']
do_standard=['TotalCharges', 'MonthlyCharges', 'tenure']
df=df.rename(columns={'Yes':'churn_yes'})

X=df[encode+do_standard]
Y=df['churn_yes']
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42,stratify=Y)
num_trans=StandardScaler()
cat_encode=OneHotEncoder(handle_unknown='ignore')
preprocess=ColumnTransformer(transformers=[('num',num_trans,do_standard),('cat',cat_encode,encode)])
learned_transformed_xtrain=preprocess.fit_transform(x_train)
trans_xtest=preprocess.transform(x_test)
model=LogisticRegression()
model.fit(learned_transformed_xtrain,y_train)
y_pred=model.predict(trans_xtest)
cm=confusion_matrix(y_test,y_pred)
display=ConfusionMatrixDisplay(cm,display_labels=['stayed','churned'])
'''display.plot(cmap='Blues')
plt.savefig("confusion matrix")
plt.show()'''
#print(classification_report(y_test,y_pred))
rf_model=RandomForestClassifier(n_estimators=100,random_state=42)
rf_model.fit(learned_transformed_xtrain,y_train)
y_pred_rf=rf_model.predict(trans_xtest)
#print(y_pred_rf[:10])
#print(accuracy_score(y_test,y_pred_rf))
#print(classification_report(y_test,y_pred_rf))
#print(rf_model.predict_proba(learned_transformed_xtrain))
gb_model=GradientBoostingClassifier(random_state=42)
gb_model.fit(learned_transformed_xtrain,y_train)
gb_model_pred=gb_model.predict(trans_xtest)
#print(classification_report(y_test,gb_model_pred))
feature_names=preprocess.get_feature_names_out()
ls_model_coef=model.coef_[0]
coef_df=pd.DataFrame({'feature':feature_names,'coefficient':ls_model_coef})
coef_df=coef_df.sort_values('coefficient',ascending=False)
#print(coef_df)
top_fea=coef_df.head(10)
bot_fea=coef_df.tail(10)
imp_fea=pd.concat([top_fea,bot_fea])
'''plt.bar(imp_fea['feature'],imp_fea['coefficient'])
plt.xticks(rotation=90)
plt.show()'''
mag_model=model.predict_proba(trans_xtest)[:,1]
mag_035=(mag_model >=0.35).astype(int)
#print("0.35\n",classification_report(y_test,mag_035))
mag_030=(model.predict_proba(trans_xtest)[:,1]>=0.30).astype(int)
#print("0.30\n",classification_report(y_test,mag_030))
mag_045=(model.predict_proba(trans_xtest)[:,1]>=0.45).astype(int)
#print("0.45\n",classification_report(y_test,mag_045))
mag_040=(model.predict_proba(trans_xtest)[:,1]>=0.40).astype(int)
#print("0.40\n",classification_report(y_test,mag_040))
final_cm=confusion_matrix(y_test,mag_035)
#print(final_cm)
display_final_cm=ConfusionMatrixDisplay(final_cm,display_labels=["stayed","churned"])
'''display_final_cm.plot(cmap='Blues')
plt.savefig("final_confusion_matrix_for_best_threshold")
plt.show()'''
final_model=Pipeline([('preprocess',preprocess),('model',LogisticRegression())])
final_model.fit(x_train,y_train)
final_prob=final_model.predict_proba(x_test)[:,1]
final_predict=(final_prob>=0.35).astype(int)
#print(classification_report(y_test,final_predict))
#print(roc_auc_score(y_test,final_prob))
fpr,trp,threshold=roc_curve(y_test,final_prob)
'''plt.plot(fpr,trp,label='Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig("roc_curve")
plt.show()'''
scores = cross_val_score(final_model,x_train,y_train,cv=5,scoring='roc_auc')
'''print(scores)
print(scores.mean())
'''
balanced_model=Pipeline([('preprocess',preprocess),('model',LogisticRegression(class_weight='balanced'))])
balanced_model.fit(x_train,y_train)
balanced_model_pred=balanced_model.predict(x_test)
#print(classification_report(y_test,balanced_model_pred))
preprocesing=final_model.named_steps['preprocess']
x_test_transformed =preprocesing.transform(x_test)
real_model=final_model.named_steps['model']
explainer=shap.LinearExplainer(real_model,x_test_transformed)
shap_values=explainer.shap_values(x_test_transformed)
#print(shap_values)
'''shap.summary_plot(shap_values,x_test_transformed,feature_names=feature_names)
plt.savefig("shap")'''
joblib.dump(final_model,'churn_model.pkl')
loaded_model=joblib.load('churn_model.pkl')

