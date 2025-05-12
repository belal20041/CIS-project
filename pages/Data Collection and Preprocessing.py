import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
import os
import uuid
from io import BytesIO

st.set_page_config(page_title="Milestone 1: Data Collection, Exploration, Preprocessing", layout="wide")
st.markdown("<h1 style='text-align: center;'>Data Collection & Preprocessing</h1>", unsafe_allow_html=True)

def load_data(file, dataset_type="train"):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.name.endswith('.parquet'):
        df = pd.read_parquet(file)
    else:
        st.error("Unsupported format. Use CSV or Parquet.")
        return None
    df = df.loc[:, ~df.columns.str.contains('^unnamed')]
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")
    return df

def detect_column_types(df, date_col=None):
    numeric = df.select_dtypes(include=['int64','float64','int32','float32']).columns.tolist()
    categorical = [c for c in df.columns if c!=date_col and (df[c].dtype in ['object','category','bool'] or df[c].nunique()/len(df)<0.05)]
    if date_col:
        numeric = [c for c in numeric if c!=date_col]
        categorical = [c for c in categorical if c!=date_col]
    return numeric, categorical

def apply_column_type_changes(df, changes, date_col=None):
    df2 = df.copy()
    new_num,new_cat=[],[]
    for col,t in changes.items():
        if col==date_col and t!='date':
            st.warning(f"Cannot change date column {col} to {t}.")
            continue
        try:
            if t=='numeric': df2[col]=pd.to_numeric(df2[col],errors='coerce'); new_num.append(col)
            elif t=='categorical': df2[col]=df2[col].astype('category'); new_cat.append(col)
            elif t=='date': df2[col]=pd.to_datetime(df2[col],errors='coerce'); date_col= date_col or col
        except Exception as e:
            st.error(f"Failed to convert {col} to {t}: {e}")
    num,cat=detect_column_types(df2,date_col)
    num=list(set(num+new_num)); cat=list(set(cat+new_cat))
    if date_col:
        num=[c for c in num if c!=date_col]
        cat=[c for c in cat if c!=date_col]
    return df2,num,cat,date_col

def clean_text_columns(df, cols):
    df2=df.copy()
    def clean(x): return x.strip().capitalize() if isinstance(x,str) else x
    for c in cols:
        if c in df2: df2[c]=df2[c].apply(clean)
    corr={"fundacion de guayaquil-1":"Fundacion de guayaquil","santo domingo de los tsachilas":"Santo domingo"}
    for c in ['description','locale_name']:
        if c in df2: df2[c]=df2[c].apply(lambda x: corr.get(x,x))
    return df2

def explore_data(df, date_col=None, num=None, cat=None, dtp="train"):
    st.markdown(f"### {dtp.capitalize()} Data Insights")
    c1,c2=st.columns(2)
    with c1:
        st.write("**Shape**:", df.shape)
        st.write("**Missing**:", df.isna().sum().to_dict())
        st.write("**Duplicates**:", df.duplicated().sum())
    with c2:
        st.write("**Types**:")
        st.dataframe(df.dtypes.reset_index().rename(columns={0:'Type','index':'Column'}))
        st.write("**Uniques**:", df.nunique().to_dict())
    st.markdown("**Missing Matrix**")
    fig,ax=plt.subplots(figsize=(10,4)); msno.matrix(df,ax=ax); st.pyplot(fig)
    sns.set_style("whitegrid")
    if date_col in df.columns:
        try:
            df[date_col]=pd.to_datetime(df[date_col],errors='coerce')
            if 'sales' in df:
                st.markdown("**Sales Trends**"); fig,ax=plt.subplots(figsize=(8,4)); df.groupby(date_col)['sales'].sum().plot(ax=ax); plt.xticks(rotation=45); st.pyplot(fig)
                st.markdown("**Monthly Seasonality**"); df['month']=df[date_col].dt.month; fig,ax=plt.subplots(figsize=(8,4)); sns.boxplot(x='month',y='sales',data=df,ax=ax); st.pyplot(fig)
        except: st.warning(f"Cannot process {date_col} for trends.")
    for col in ['family','city','state','store_nbr']:
        if col in df and 'sales' in df:
            st.markdown(f"**Sales by {col.capitalize()}**"); fig,ax=plt.subplots(figsize=(12,4)); sns.boxplot(data=df,x=col,y='sales',ax=ax); plt.xticks(rotation=45); st.pyplot(fig)
    if 'sales' in df and 'onpromotion' in df:
        st.markdown("**Sales vs Promotions**"); fig,ax=plt.subplots(figsize=(8,4)); sns.scatterplot(x='onpromotion',y='sales',data=df,ax=ax); st.pyplot(fig)
    if 'transferred' in df:
        st.markdown("**Holiday Distribution**"); st.dataframe(df['transferred'].value_counts())
    if 'sales' in df:
        st.markdown("**Sales Distribution**"); fig,ax=plt.subplots(figsize=(8,4)); sns.histplot(df['sales'],bins=30,kde=True,ax=ax); st.pyplot(fig)
    if num:
        st.markdown("**Correlation Heatmap**"); fig,ax=plt.subplots(figsize=(8,6)); sns.heatmap(df[num].corr(),annot=True,fmt=".2f",linewidths=0.5,ax=ax); st.pyplot(fig)
        vals=[c for c in num if pd.api.types.is_numeric_dtype(df[c])]
        if vals:
            st.markdown("**Outliers**"); fig,axes=plt.subplots(1,len(vals),figsize=(4*len(vals),4)); axes=[axes] if len(vals)==1 else axes
            for i,c in enumerate(vals): sns.boxplot(x=df[c],ax=axes[i]); axes[i].set_title(c)
            plt.tight_layout(); st.pyplot(fig)
            for c in vals:
                data=df[c].dropna(); Q1,Q3=data.quantile(0.25),data.quantile(0.75)
                IQR=Q3-Q1; lb,ub=Q1-1.5*IQR,Q3+1.5*IQR
                iqr_out=df[(df[c]<lb)|(df[c]>ub)][c]
                z_out=df.loc[data.index][abs(zscore(data))>3][c] if data.std()>1e-6 else pd.Series()
                st.write(f"{c}: IQR Outliers={len(iqr_out)}, Z-score={len(z_out)}")
def preprocess_data(df, num=None, cat=None, date_col=None, handle_outliers='remove', normalize=False, dtp="train"):
    df2=df.copy()
    for c in df2:
        if df2[c].isna().sum()>0:
            if num and c in num: df2[c].fillna(df2[c].median(),inplace=True)
            elif cat and c in cat: df2[c].fillna(df2[c].mode()[0],inplace=True)
            else: df2[c].fillna('Unknown',inplace=True)
    init_rows=df2.shape[0]
    df2=df2.drop_duplicates()
    if num and handle_outliers:
        for c in num:
            if c in df2:
                Q1,Q3=df2[c].quantile(0.25),df2[c].quantile(0.75); IQR=Q3-Q1
                lb,ub=Q1-1.5*IQR,Q3+1.5*IQR
                if handle_outliers=='remove': df2=df2[(df2[c]>=lb)&(df2[c]<=ub)]
                else: df2[c]=df2[c].apply(lambda x: df2[c].median() if x<lb or x>ub else x)
    if date_col in df2 and 'year' not in df2:
        try:
            df2[date_col]=pd.to_datetime(df2[date_col],errors='coerce')
            df2['day']=df2[date_col].dt.day; df2['month']=df2[date_col].dt.month; df2['year']=df2[date_col].dt.year
            df2['dayofweek']=df2[date_col].dt.dayofweek; df2['sin_month']=np.sin(2*np.pi*df2['month']/12)
            df2['week']=df2[date_col].dt.isocalendar().week; df2['is_weekend']=df2[date_col].dt.weekday.apply(lambda x:1 if x>=5 else 0)
            df2['season']=pd.cut(df2['month'],bins=[0,3,6,9,12],labels=['Q1','Q2','Q3','Q4'])
        except: st.warning(f"Cannot process time features for {date_col}.")
    if normalize and num:
        valid=[c for c in num if c in df2 and df2[c].std()>1e-6]
        if valid:
            scaler=MinMaxScaler()
            df2[valid]=scaler.fit_transform(df2[valid]) if dtp=='train' else scaler.transform(df2[valid])
    df2=clean_text_columns(df2, ["family","city","state","cluster","type_y","locale","locale_name","description","transferred"])
    return df2, init_rows-df2.shape[0]
def engineer_features(train,test,num,cat,target='sales'):
    tr,te=train.copy(),test.copy()
    if target in tr and 'onpromotion' in tr:
        tr['sales_onpromo']=tr[target]*tr['onpromotion']
        te['sales_onpromo']=te['onpromotion']*0 if 'onpromotion' in te else 0
    if 'onpromotion' in tr and 'is_weekend' in tr:
        tr['promo_weekend']=tr['onpromotion']*tr['is_weekend']
        te['promo_weekend']=te['onpromotion']*te['is_weekend'] if 'onpromotion' in te and 'is_weekend' in te else 0
    for col in ['city','state']:
        if col in tr:
            m=tr.groupby(col)[target].mean().to_dict(); tr[f'{col}_encoded']=tr[col].map(m)
            te[f'{col}_encoded']=te[col].map(m).fillna(tr[target].mean())
    for col in ['family','locale_name']:
        if col in tr:
            f=tr[col].value_counts(normalize=True).to_dict(); tr[f'{col}_encoded']=tr[col].map(f)
            te[f'{col}_encoded']=te[col].map(f).fillna(0)
    if 'store_nbr' in tr and 'month' in tr and target in tr:
        tr['avg_sales_store_month']=tr.groupby(['store_nbr','month'])[target].transform('mean')
        d=tr.groupby(['store_nbr','month'])[target].mean().to_dict()
        te['avg_sales_store_month']=te.apply(lambda x:d.get((x['store_nbr'],x['month']),tr[target].mean()),axis=1)
    for df in [tr,te]:
        if 'description' in df: df['is_holiday']=df['description'].str.contains('Holiday|Navidad',case=False,na=False).astype(int)
        for c in ['transferred','is_weekend','is_holiday']:
            if c in df: df[c]=df[c].astype(int)
        orders={'National':2,'Regional':1,'Local':0}
        if 'locale' in df: df['locale_encoded']=df['locale'].map(orders).fillna(0)
        orders2={'Holiday':2,'Event':1,'Bridge':0}
        if 'type_y' in df: df['type_y_encoded']=df['type_y'].map(orders2).fillna(0)
    if 'dcoilwtico' in tr:
        q25,q75=tr['dcoilwtico'].quantile(0.25),tr['dcoilwtico'].quantile(0.75)
        bins=([-np.inf,q25,np.inf] if q25==q75 else [-np.inf,q25,q75,np.inf])
        labels=(['low','high'] if q25==q75 else ['low','medium','high'])
        tr['dcoilwtico_bin']=pd.cut(tr['dcoilwtico'],bins=bins,labels=labels)
        te['dcoilwtico_bin']=pd.cut(te['dcoilwtico'],bins=bins,labels=labels) if 'dcoilwtico' in te else None
    return tr,te

def get_download_file(df, filename):
    buf=BytesIO(); df.to_csv(buf,index=False); buf.seek(0)
    return buf.getvalue(),'text/csv'

def main():
    st.divider()
    with st.expander("ℹ️ Project Info"):
        st.markdown("**Objective**: Collect, explore, and preprocess historical sales data.")
        st.write(["Data Collection", "Data Exploration", "Data Preprocessing", "Feature Engineering"])
    t_tab,tt_tab=st.tabs(["Train Data","Test Data"])
    with t_tab:
        uf=st.file_uploader("Upload Train Data",type=['csv','parquet'],key="train")
        if uf:
            df=load_data(uf)
            if df is not None:
                st.session_state['train_df']=df; st.dataframe(df.head(),height=150)
                with st.form("train_cfg"):
                    dc=st.selectbox("Date Column",['None']+list(df.columns)); dc=None if dc=='None' else dc
                    tc=st.selectbox("Target",df.columns,index=df.columns.tolist().index('sales') if 'sales' in df else 0)
                    num,cat=detect_column_types(df,dc)
                    with st.expander("Adjust Column Types"): pass
                    om=st.selectbox("Outliers",['None','Remove','Replace'],key="train_o"); om=om.lower() if om!='None' else None
                    nz=st.checkbox("Normalize")
                    st.form_submit_button("Apply",on_click=lambda: st.session_state.update({
                        'train_date_col':dc,'train_target_col':tc,'train_numeric_cols':num,'train_categorical_cols':cat,'train_outlier_method':om,'train_normalize':nz
                    }))
                with st.form("train_proc"):
                    st.markdown("**Process Train Data**"); submitted=st.form_submit_button("Run")
                    if submitted:
                        df2=apply_column_type_changes(st.session_state['train_df'],st.session_state['train_type_changes'],st.session_state['train_date_col'])[0]
                        explore_data(df2,st.session_state['train_date_col'],st.session_state['train_numeric_cols'],st.session_state['train_categorical_cols'])
                        proc,rem=preprocess_data(df2,st.session_state['train_numeric_cols'],st.session_state['train_categorical_cols'],st.session_state['train_date_col'],st.session_state['train_outlier_method'],st.session_state['train_normalize'])
                        st.session_state['processed_train']=proc; st.write(f"Processed: {rem} duplicates removed, {proc.shape[0]} rows remain"); st.dataframe(proc.head(),height=150)
    with tt_tab:
        pass
    st.divider(); st.markdown("**Created with Belal Khamis**")
if __name__=="__main__": main()
