
# coding: utf-8

# In[1]:

import os


# In[12]:

paths = sorted([fn for fn in os.listdir('test_stg1')])

fw_out = open('submission_random.csv', 'w')
fw_out.write('image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT\n')
for i, img in enumerate(paths):
    pred = ['0.125']*8
    fw_out.write('%s,%s\n' % (img, ','.join(pred)))


# In[ ]:



