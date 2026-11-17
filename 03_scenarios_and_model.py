import pandas as pd, numpy as np
pd.set_option('display.width',250,'display.max_columns',50,'display.max_rows',100)
S='work/'
d=pd.read_pickle(S+'d2.pkl')
m=d[d.arr.dt.month<=8]
# mix vs rate decomposition 2016->2017 (cancel rate, bookings-weighted)
a=m[m.arrival_date_year==2016].groupby('market_segment').is_canceled.agg(['size','mean']); b=m[m.arrival_date_year==2017].groupby('market_segment').is_canceled.agg(['size','mean'])
a['w']=a['size']/a['size'].sum(); b['w']=b['size']/b['size'].sum()
j=a.join(b,lsuffix='16',rsuffix='17',how='inner')
mix=((j.w17-j.w16)*j.mean16).sum(); rate=(j.w17*(j.mean17-j.mean16)).sum()
print('cr16',(j.w16*j.mean16).sum(),'cr17',(j.w17*j.mean17).sum(),'mix',mix,'rate',rate)
print(j[['mean16','mean17','w16','w17']].round(3))
ot=m[(m.market_segment=='Online TA')]
print(ot.groupby(['arrival_date_year','deposit_type']).is_canceled.agg(['size','mean']))
print(ot.groupby('arrival_date_year').agg(cr=('is_canceled','mean'),lt=('lead_time','mean'),adr=('adr','mean')))
# scenarios (on cleaned data as-is)
tot_lost=d.lost_revenue.sum(); nd=d[d.deposit_type=='No Deposit']
o=d[(d.market_segment=='Online TA')&(d.deposit_type=='No Deposit')].copy()
o['ltb']=pd.cut(o.lead_time,[-1,30,90,180,1000],labels=['0-30','31-90','91-180','181+'])
g=o.groupby(['hotel','ltb'],observed=True).agg(cr=('is_canceled','mean'),lost=('lost_revenue','sum')).reset_index()
bench=g[g.ltb=='31-90'].set_index('hotel').cr
g['newlost']=np.where(g.ltb.isin(['91-180','181+']),g.lost*g.hotel.map(bench)/g.cr,g.lost)
A=(g.lost-g.newlost).sum(); print('A: long-lead OTA to 31-90 benchmark',A, A/tot_lost)
ol=o.lost_revenue.sum(); print('OTA no-dep lost',ol)
for p in (.1,.2,.3): print('B',p,ol*p)
ndc=d[(d.is_canceled==1)&(d.deposit_type=='No Deposit')]
late=ndc[(ndc.arr-ndc.reservation_status_date).dt.days<=30].lost_revenue.sum(); print('late nodep cancels <=30d',late)
for p in (.1,.2,.3): print('C resell',p,late*p)
# Direct-rate benchmark: if Online TA no-dep cancel rate matched Offline TA no-dep (17%)
print('D', ol*(1-0.172/o.is_canceled.mean()), o.is_canceled.mean())
# prediction (pre-booking features only)
try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    X=d[['hotel','lead_time','arrival_date_month','arrival_date_week_number','stays_in_weekend_nights','stays_in_week_nights','adults','children','babies','meal','country','market_segment','distribution_channel','is_repeated_guest','previous_cancellations','previous_bookings_not_canceled','reserved_room_type','deposit_type','agent','days_in_waiting_list','customer_type','adr']].copy()
    for c in X.select_dtypes('object'): X[c]=X[c].astype('category')
    X['agent']=X.agent.fillna(-1)
    y=d.is_canceled
    tr=d.arrival_date_year<2017  # time split: train on pre-2017, test 2017
    mdl=HistGradientBoostingClassifier(categorical_features='from_dtype',max_iter=200,random_state=0).fit(X[tr],y[tr])
    p=mdl.predict_proba(X[~tr])[:,1]; print('AUC time-split pre-booking',roc_auc_score(y[~tr],p))
    X2=X.drop(columns=['deposit_type']); mdl2=HistGradientBoostingClassifier(categorical_features='from_dtype',max_iter=200,random_state=0).fit(X2[tr],y[tr])
    print('AUC without deposit_type',roc_auc_score(y[~tr],mdl2.predict_proba(X2[~tr])[:,1]))
    # capture: top decile risk share of lost rev in 2017
    t=d[~tr].assign(p=p); t['dec']=pd.qcut(t.p.rank(method='first'),10,labels=False)
    r=t.groupby('dec').agg(cr=('is_canceled','mean'),lost=('lost_revenue','sum')); r['share']=r.lost/r.lost.sum(); print(r.round(3))
except Exception as e: print('sk err',e)
