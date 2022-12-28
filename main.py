from pulp import *
import pandas as pd
import numpy as np

model = LpProblem("Nurse-Scheduling-Problem", LpMinimize)

# Definition of parameters
number_nurses = 2
number_days_schedule = 31
shifts_per_day = 3
number_nurses_per_shift = 4
min_number_nurses_per_shift = 2
max_number_shifts_per_days_scheduled = round(number_days_schedule / 7 * 5)
min_break_between_two_shifts = 2  # Min 2 Shifts Break between shifts
max_consecutive_days_with_shifts = 5
days_weekend = 2

#Add vacation days

# Creation of Availibility Plan -> 1 if available else zero
availability_plan = np.ones((number_nurses, number_days_schedule, shifts_per_day))

#Function to set values in availibility Plan to zero based on nurse and day

def vaccation_request(nurse, day):
    for k in range(shifts_per_day):
        availability_plan[nurse, day, k] = 0

# Vaccation requests
vaccation_request(1, 1)
vaccation_request(1, 2)
vaccation_request(1, 3)

#3D Array of nurse schedule.
# First dimension: Nurse
# Second Dimension: Day
# Third Dimension: Shift

# Value = 0 if nurse is not working in a shift and 1 if nurse is working
# Can only be either zero or one
nurses = LpVariable.matrix("Nurse", (range(number_nurses), range(number_days_schedule), range(shifts_per_day)),
                           cat="Integer", lowBound=0, upBound=1)

#Model is able to break certain rules but is penatalized for it.
# The model is incetivized to not break rules if possible

# Penalties
#Penalty for not enough nurses in a shift
penalty_nurses_per_shift = LpVariable.matrix("penalty_nurses_per_shift",
                                             (range(number_days_schedule), range(shifts_per_day)), lowBound=0)
# Penalty if one nurse works more than 5 days in a row
penalty_to_many_consecutive_days_working = LpVariable.matrix("penalty_to_many_consecutive_days_working",
                                                             (range(number_nurses),range(number_days_schedule)), lowBound=0)
# Penalty if one nurse works more shifts in the schedule than the maximum
penalty_to_many_days_working = LpVariable.matrix("penalty_to_many_days_working", (range(number_nurses)), lowBound=0)
# Penalty if one nurse needs to work during vacation
penalty_vacation = LpVariable.matrix("penalty_vacation",
                                     (range(number_nurses), range(number_days_schedule), range(shifts_per_day)),
                                     lowBound=0)
penalty_min_nurses_per_shift = LpVariable.matrix("penalty_min_nurses_per_shift",
                                             (range(number_days_schedule), range(shifts_per_day)), lowBound=0)

# Weighting of Penalties:

# Each Penalty receives a weight since some rule breaks are more serious than others
# penalties
factor_penalty_nurses_per_shift = 200
factor_penalty_to_many_days_working = 400
factor_penalty_to_many_consecutive_days_working = 100
factor_penalty_vacation = 1200
factor_penalty_min_nurses_per_shift = 2400



# Objective Function is the function that the model minimizes
# This function includes all penalties multiplied with the respetive factor
obj_func = lpSum(penalty_min_nurses_per_shift) * factor_penalty_min_nurses_per_shift + lpSum(penalty_vacation) * factor_penalty_vacation + lpSum(penalty_to_many_days_working) * factor_penalty_to_many_days_working + lpSum(penalty_nurses_per_shift) * factor_penalty_nurses_per_shift + lpSum(penalty_to_many_consecutive_days_working) * factor_penalty_to_many_consecutive_days_working

# Add Function to Model
model += obj_func


# Constraint 1: Number nurses per shift
# Ideally number_nurses_per_shift but can deviate to min_number_nurses_per_shift to give the model more flexibility if ideal
# nurses per shift cannot be met. Model is penalized if it is below the ideal number

# Iterate through Days and Shifts and take the sum over nurses (e.g., Count how many nurses are working in a shift)
for j in range(number_days_schedule):
    for k in range(shifts_per_day):
        # Add Constraint that allows the model to have nurses working than ideal. The difference between ideal and actual
        # value is saved in the penalty
        model += lpSum(nurses[i][j][k] for i in range(number_nurses)) + penalty_nurses_per_shift[j][
            k] >= number_nurses_per_shift
        # The model is heavily penalized if it goes below the minimum number but it is still possible if no other solution is possible
        model += lpSum(nurses[i][j][k] for i in range(number_nurses)) +  penalty_min_nurses_per_shift[j][k] >= min_number_nurses_per_shift

# Constraint 2: Days without break
# Nurses need at least one day off after reaching max_consecutive_days_with_shifts. T
# The model can deviate but will be penalized

for i in range(number_nurses):
    # Constraint: Max Number of Shifts in the Schedule
    model += lpSum(nurses[i]) - penalty_to_many_days_working[i] <= max_number_shifts_per_days_scheduled
    for j in range(max_consecutive_days_with_shifts + 1, number_days_schedule):
        # Constraint: Consecutive shifts
        model += lpSum(nurses[i][j - max_consecutive_days_with_shifts:j + 1]) - \
                 penalty_to_many_consecutive_days_working[
            i][j - max_consecutive_days_with_shifts:j + 1] <= max_consecutive_days_with_shifts

# Constraint 4: Vacation
# Nurses who are on vacation can not work on that day

for i in range(number_nurses):
    for j in range(number_days_schedule):
        for k in range(shifts_per_day):
            model += lpSum(nurses[i][j][k]) - penalty_vacation[i][j][k] <= availability_plan[i][j][k]
#
# Constraint 5: Min Break between two Shifts
nurses_2D = LpVariable.matrix("Nurse", (range(number_nurses), range(number_days_schedule * shifts_per_day)),
                              cat="Integer", lowBound=0, upBound=1)

# Make Sure that 2D (Nurse, Shifts) and 3D representation (Nurse, Days, Shifts) are equal
for i in range(number_nurses):
    for j in range(number_days_schedule):
        for k in range(shifts_per_day):
            model += lpSum(nurses[i][j][k]) == lpSum(nurses_2D[i][j * shifts_per_day + k])

# Add Constraint on 2D model
for i in range(number_nurses):
    for j in range(min_break_between_two_shifts, number_days_schedule * shifts_per_day):
        # Constraint to have maximum 1 Shift within a 3 Shift time frame
        model += lpSum(nurses_2D[i][j - min_break_between_two_shifts:j + 1]) <= 1

# solve optimization model
model.solve(PULP_CBC_CMD())
status = LpStatus[model.status]

print(status)
print("Total Cost:", model.objective.value())

print(lpSum(nurses))

model.writeLP("Nurse-Scheduling-Problem.lp")


#Print Schedule per Shift
for j in range(number_days_schedule):
    print("Working on Day: " + str(j))
    for k in range(shifts_per_day):
        print("Working in Shift: " + str(k))
        for i in range(number_nurses):
            if (nurses[i][j][k].varValue == 1):
                print("Nurse "+str(i))


#Print Schedule per Nurse
for i in range(number_nurses):
    print("Nurse: " + str(i)+":")
    for j in range(number_days_schedule):
        for k in range(shifts_per_day):
            if (nurses[i][j][k].varValue == 1):
                print("Day "+str(j)+" Shift "+str(k))


# for i in range(len(nurses_2D[11])) :
#     print("Shift "+ str(i)+": " +str(nurses_2D[11][i].varValue))

print(status)
print("Total Cost:", model.objective.value())
