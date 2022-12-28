#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pulp import *
import pandas as pd
import numpy as np

# In[2]:


model = LpProblem("Nurse_scheduling_problem", LpMinimize)

number_nurses = 10
number_days = 7
shifts_per_day = 3
number_nurses_per_shift = 4

nurses = LpVariable.matrix("dec_var", (range(number_nurses), range(number_days), range(shifts_per_day)), 0, 1,
                           cat='Binary')

optimizationVariable = LpVariable("optVar", lowBound=0)
obj_func = lpSum(optimizationVariable)

# In[3]:


for j in range(number_days):
    for k in range(shifts_per_day):
        # print(lpSum(nurses[i][j][k] for i in range(number_nurses)) == number_nurses_per_shift)
        model += lpSum(nurses[i][j][k] for i in range(number_nurses)) == number_nurses_per_shift

# In[4]:


for i in range(number_nurses):
    for j in range(number_days):
        # print(lpSum(nurses[i][j][k] for k in range(shifts_per_day)) <= 1)
        model += lpSum(nurses[i][j][k] for k in range(shifts_per_day)) <= 1

# In[5]:


for i in range(number_nurses):
    # print(lpSum(nurses[i]) - optimizationVariable  <= 5)
    model += lpSum(nurses[i]) - optimizationVariable <= 5

# In[6]:


for j in range(number_days):
    for k in range(shifts_per_day):
        # print(lpSum(lpSum(nurses[i][j][k] - number_nurses_per_shift for i in range(number_nurses) )) == optimizationVariable)

        model += number_nurses_per_shift - lpSum(nurses[i][j][k] for i in range(number_nurses)) == optimizationVariable

    # In[10]:

for i in range(number_nurses):
    for j in range(number_days):
        for k in range(shifts_per_day):
            model += nurses[i][j][k] <= 1
            model += nurses[i][j][k] >= 0
            print(nurses[i][j][k] >= 0)

# In[ ]:


# In[8]:


# model.solve()
model.solve(PULP_CBC_CMD())

status = LpStatus[model.status]

print(status)

# In[9]:


for v in model.variables():
    try:
        print(v.name, "=", v.value())
    except:
        print("error couldnt find value")

# In[11]:


print(model.constraints)

