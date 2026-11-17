import pandas as pd, numpy as np
S='work/'
d=pd.read_pickle(S+'d2.pkl')
d['nights']=d.n
d['arrival_month']=d.arr.dt.to_period('M').dt.to_timestamp()
d['lead_time_bucket']=pd.cut(d.lead_time,[-1,7,30,90,180,365,2000],labels=['0-7','8-30','31-90','91-180','181-365','366+']).astype(str)
d['days_cancelled_before_arrival']=np.where(d.reservation_status=='Canceled',(d.arr-d.reservation_status_date).dt.days,np.nan)
d['cancel_notice_bucket']=pd.cut(d.days_cancelled_before_arrival,[-1,7,30,90,2000],labels=['0-7d','8-30d','31-90d','90d+']).astype(str).replace('nan','')
d['is_duplicate_row']=d.drop(columns=['arr','n','nights','arrival_month','lead_time_bucket','days_cancelled_before_arrival','cancel_notice_bucket']).duplicated().astype(int)
d['flag_zero_night']=(d.nights==0).astype(int); d['flag_adr_zero']=(d.adr==0).astype(int); d['flag_adr_outlier']=(d.adr>1000).astype(int)
d['non_refund_flag']=(d.deposit_type=='Non Refund').astype(int)
d.rename(columns={'arr':'arrival_date'}).drop(columns=['n']).to_csv('output/hotel_revenue_tableau_ready.csv',index=False)
def g(by,df=d):
    t=df.groupby(by,observed=True).agg(bookings=('is_canceled','size'),cancel_rate=('is_canceled','mean'),expected=('expected_revenue','sum'),realized=('realized_revenue','sum'),lost=('lost_revenue','sum'))
    t['lost_pct_of_expected']=t.lost/t.expected; t['share_of_total_lost']=t.lost/df.lost_revenue.sum(); return t.sort_values('lost',ascending=False).reset_index()
nd=d[d.deposit_type=='No Deposit']
sc=pd.DataFrame([
 ['Baseline lost revenue (all, as in your file)',d.lost_revenue.sum(),'Assumes $0 recovered on every cancellation/no-show'],
 ['  of which Non-Refund deposit bookings',d[d.deposit_type=='Non Refund'].lost_revenue.sum(),'Likely overstated - hotel may have kept deposits'],
 ['  of which No Deposit (addressable pool)',nd.lost_revenue.sum(),''],
 ['  of which Online TA, No Deposit',nd[nd.market_segment=='Online TA'].lost_revenue.sum(),'Largest single pocket'],
 ['A: Online TA long-lead (91+d) cancel rate falls to own 31-90d rate',1004877,'Per-hotel internal benchmark'],
 ['B: Online TA no-deposit cancellations down 10% (relative)',1020080,'Illustrative'],
 ['B: ... down 20%',2040159,'Illustrative'],
 ['B: ... down 30%',3060239,'Illustrative'],
 ['C: Resell 20% of No-Deposit cancellations made <=30d before arrival',1028047,'Pool = $5.14M; resale rate is an assumption'],
 ['D: Online TA no-deposit cancel rate falls to Offline TA level (17.2%)',5412663,'Upper bound, not a realistic target'],
],columns=['Scenario','Revenue','Note'])
with pd.ExcelWriter('output/revenue_leakage_analysis_tables.xlsx') as w:
    sc.to_excel(w,sheet_name='scenarios',index=False)
    for n,b in [('by_hotel','hotel'),('by_segment','market_segment'),('by_deposit','deposit_type'),('by_leadtime','lead_time_bucket'),('by_cancel_notice','cancel_notice_bucket'),('by_country_top','country'),('by_agent_top','agent'),('by_month','arrival_month'),('segment_x_hotel',['market_segment','hotel']),('segment_x_deposit',['market_segment','deposit_type']),('by_special_requests','total_of_special_requests'),('by_customer_type','customer_type')]:
        t=g(b); (t.head(25) if n.endswith('top') else t).to_excel(w,sheet_name=n,index=False)
print('ok')
