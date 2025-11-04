import pandas as pd
from pyBKT.models import Model
df = pd.DataFrame({
  'order_id':  range(10),
  'user_id':   ['u1']*10,
  'skill_name':['add']*10,
  'correct':   [0,0,1,0,1,1,1,0,1,1],
})
m = Model(seed=42, num_fits=1)
m.fit(data=df, defaults={'order_id':'order_id','user_id':'user_id','skill_name':'skill_name','correct':'correct'})
print("AUC:", m.evaluate(data=df, metric='auc'))
print(m.predict(data=df).head())