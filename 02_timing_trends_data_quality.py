import pandas as pd, numpy as np
pd.set_option('display.width',250,'display.max_columns',50,'display.max_rows',100)
S='work/'
d=pd.read_pickle(S+'d.pkl')
mm={m:i for i,m in enumerate(['January','February','March','April','May','June','July','August','September','October','November','December'],1)}
d['arr']=pd.to_datetime(dict(year=d.arrival_date_year,month=d.arrival_date_month.map(mm),day=d.arrival_date_day_of_month))
d['n']=d.stays_in_weekend_nights+d.stays_in_week_nights
c=d[d.reservation_status=='Canceled'].copy()
c['days_before']=(c.arr-c.reservation_status_date).dt.days
print(c.days_before.describe())
print((c.days_before<0).sum())
c['tb']=pd.cut(c.days_before,[-1000,-1,7,30,90,1000],labels=['after-arrival?','0-7d','8-30d','31-90d','90d+'])
t=c.groupby('tb',observed=True).lost_revenue.agg(['size','sum']);t['pct']=t['sum']/t['sum'].sum();print(t.round(3))
t=c.groupby(['market_segment','tb'],observed=True).lost_revenue.sum().unstack().round(0);print(t)
# no-deposit only
nd=c[c.deposit_type=='No Deposit']
t=nd.groupby('tb',observed=True).lost_revenue.agg(['size','sum']);t['pct']=t['sum']/t['sum'].sum();print('NoDeposit',t.round(3))
# Online TA no deposit lead time x hotel
o=d[(d.market_segment=='Online TA')&(d.deposit_type=='No Deposit')].copy()
o['lt']=pd.cut(o.lead_time,[-1,30,90,180,1000],labels=['0-30','31-90','91-180','181+'])
t=o.groupby(['hotel','lt'],observed=True).agg(b=('is_canceled','size'),cr=('is_canceled','mean'),exp=('expected_revenue','sum'),lost=('lost_revenue','sum'));print(t.round(3))
# year trend matched months Jan-Aug
m=d[d.arr.dt.month<=8]
print(m.groupby(['arrival_date_year','hotel']).agg(b=('is_canceled','size'),cr=('is_canceled','mean'),exp=('expected_revenue','sum'),lost=('lost_revenue','sum')).assign(lp=lambda x:x.lost/x.exp).round(3))
d['ym']=d.arr.dt.to_period('M')
q=d.groupby('ym').agg(b=('is_canceled','size'),cr=('is_canceled','mean'),exp=('expected_revenue','sum'),lost=('lost_revenue','sum'))
print(q.round(3))
# segment mix shift
print(m.groupby(['arrival_date_year','market_segment']).size().unstack(0).pipe(lambda x:x/x.sum()).round(3))
print(m.groupby(['arrival_date_year']).lead_time.mean())
# dedupe sensitivity
cols=[x for x in d.columns if x not in ('arr','n')]
dd=d.drop_duplicates(subset=[x for x in d.columns if x not in('arr','n','ym')])
print(len(d),len(dd),d.lost_revenue.sum()/d.expected_revenue.sum(),dd.lost_revenue.sum()/dd.expected_revenue.sum(),dd.is_canceled.mean())
print(dd.groupby('market_segment').agg(cr=('is_canceled','mean'),lp=('lost_revenue','sum'),e=('expected_revenue','sum')).assign(p=lambda x:x.lp/x.e).round(3))
# zero nights / adr0
z=d[d.n==0]; print(z.reservation_status.value_counts(), z.market_segment.value_counts().head(3))
print(d[d.adr==0].market_segment.value_counts())
print(d[d.adr>1000][['hotel','adr','n','expected_revenue','reservation_status']])
# is a non-refund in Portugal group effect: agent 1 etc
nr=d[d.deposit_type=='Non Refund']; print(nr.groupby('agent').agg(b=('is_canceled','size'),lost=('lost_revenue','sum')).sort_values('lost',ascending=False).head(5))
print(nr.groupby('lead_time').size().sort_values(ascending=False).head(3), nr.lead_time.median(), d[d.deposit_type=='No Deposit'].lead_time.median())
# Concentration: top 2 agents share of No deposit lost, segments
ndall=d[d.deposit_type=='No Deposit']; print(ndall.groupby('agent').lost_revenue.sum().sort_values(ascending=False).head(5)/ndall.lost_revenue.sum())
# Online TA by agent 9 / 240
print(d[d.market_segment=='Online TA'].groupby('agent').agg(b=('is_canceled','size'),cr=('is_canceled','mean'),lost=('lost_revenue','sum')).sort_values('lost',ascending=False).head(5))
d.to_pickle(S+'d2.pkl')
