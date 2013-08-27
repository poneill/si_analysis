import csv
from utils import *
import numpy as np
import scikits.statsmodels.api as sm
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

with open("biol141_fall12.csv") as f:
    lines = list(csv.reader(f))

(LastName, FirstName, MiddleName, EmployeeID, CampusID, MyumbcID, Email,
 Phone, Gender, Ethnicity, GPA, PriorSchoolGPA, PriorSchoolType, Term, Class,
 Section, RegistrationDate, ClassEnrollStatus, Grade, AcademicLevel,
 AdmitType, TermStatus, ReceivedLRCAlert,
 SITutorAttendanceCount) = transpose(lines[1:])

numeric_grade_dict = {'':None,'F':0,'W':None,'D':1,'I':None,'C':2,'B':3,'A':4}
numeric_grade = [numeric_grade_dict[grade] for grade in Grade]
#filter students who dropped

def got_grade(xs):
    stayed = indices_where(numeric_grade,lambda x:not x is None)
    return rslice(xs,stayed)

def convert(x):
    try:
        return float(x)
    except:
        return None

si = got_grade(map(int,SITutorAttendanceCount))
PriorSchoolGPA = got_grade(map(convert,PriorSchoolGPA))
GPA = got_grade(map(float,GPA))
grade = got_grade(numeric_grade)
Gender = got_grade(Gender)
binary_gender = [int(g=='M') for g in Gender]
males = indices_where(Gender,lambda g:g=="M")
females = indices_where(Gender,lambda g:g=="F")
male_grades = rslice(grade,males)
female_grades = rslice(grade,females)
races = got_grade(Ethnicity)
is_white = [int(r == "WHITE") for r in races]
is_black = [int(r == "BLACK") for r in races]
is_asian = [int(r == "ASIAN") for r in races]
is_hispa = [int(r == "HISPA") for r in races]
is_multi = [int(r == "MULTI") for r in races]
is_nspec = [int(r == "NSPEC") for r in races]
alerted = [a == 'Y' for a in got_grade(ReceivedLRCAlert)]
year = [{"Freshman":1,"Sophomore":2,"Junior":3,"Senior":4}[x]
        for x in got_grade(AcademicLevel)]

X = np.column_stack((GPA,si,binary_gender))
X = sm.add_constant(X,prepend=True)

gender_res = sm.OLS(grade,
             sm.add_constant(np.column_stack((GPA,si,binary_gender)),
                             prepend=True)).fit()

gpa_si_vars = sm.add_constant(np.column_stack((GPA,si)),
                             prepend=True)
res = sm.OLS(grade,
             gpa_si_vars).fit()

year_res = sm.OLS(grade,
             sm.add_constant(np.column_stack((GPA,si,alerted,year)),
                             prepend=True)).fit()

pure_year_res = sm.OLS(grade,
                       sm.add_constant(np.column_stack((year,)),
                                       prepend=True)).fit()

alerted_res = sm.OLS(grade,
                     sm.add_constant(np.column_stack((GPA,si,alerted)),
                                     prepend=True)).fit()
race_res = sm.OLS(grade,
                  sm.add_constant(np.column_stack((GPA,si,is_white,is_black,is_asian,
                                                   is_hispa,is_multi,is_nspec)),
                                  prepend=True)).fit()

gpa_vars = sm.add_constant(np.column_stack((GPA,)),
                             prepend=True)
gpa_res = sm.OLS(grade,
             gpa_vars).fit()

si_vars = sm.add_constant(np.column_stack((si,)),
                             prepend=True)
si_res = sm.OLS(grade,
             si_vars).fit()

def cross_validate_model(variables,n,randomize=False):
    obs = len(variables)
    cv_indices = cv(range(obs),n,randomize=randomize)
    mses = []
    for trial in xrange(n):
        train_indices,test_indices = cv_indices[trial]
        train_x = variables[train_indices,]
        train_grades = rslice(grade,train_indices)
        test_x = variables[test_indices,]
        test_grades = rslice(grade,test_indices)
        res = sm.OLS(train_grades,train_x).fit()
        predicted_grades = res.predict(test_x)
        mse = mean([(pred-obs)**2
                    for (pred,obs) in zip(predicted_grades,test_grades)])
        print mse
        mses.append(mse)
    return mses

def plot_si_vs_grade(filename=None):
    plt.scatter(si,grade)
    plt.xlabel("SI attendance count")
    plt.ylabel("Class Grade (on GPA scale)")
    plt.title("SI attendance vs. Grade in BIOL 141 FA12")
    print pearsonr(si,grade)
    maybesave(filename)

def plot_gpa_vs_grade(filename=None):
    plt.scatter(GPA,grade)
    plt.xlabel("Prior GPA")
    plt.ylabel("Class Grade (on GPA scale)")
    plt.title("GPA vs. Grade in BIOL 141 FA12")
    print pearsonr(GPA,grade)
    maybesave(filename)

def fit_si_vs_grade(filename=None):
    plt.scatter(si_res.fittedvalues,grade)
    plt.xlabel("Predicted Grade: SI Model")
    plt.ylabel("Class Grade (on GPA scale)")
    plt.title("Predicted vs. Actual Grade in BIOL 141 FA12 (SI Model)")
    plt.plot([0,5],[0,5])
    print pearsonr(si_res.fittedvalues,grade)
    maybesave(filename)

def fit_gpa_vs_grade(filename=None):
    plt.scatter(gpa_res.fittedvalues,grade)
    plt.xlabel("Predicted Grade: GPA Model")
    plt.ylabel("Class Grade (on GPA scale)")
    plt.title("Predicted vs. Actual Grade in BIOL 141 FA12 (GPA Model)")
    plt.plot([0,5],[0,5])
    print pearsonr(gpa_res.fittedvalues,grade)
    maybesave(filename)

def fit_si_gpa_vs_grade(filename=None):
    plt.scatter(res.fittedvalues,grade)
    plt.xlabel("Predicted Grade: SI + GPA Model")
    plt.ylabel("Class Grade (on GPA scale)")
    plt.title("Predicted vs. Actual Grade in BIOL 141 FA12 (SI + GPA Model)")
    plt.plot([0,5],[0,5])
    print pearsonr(res.fittedvalues,grade)
    maybesave(filename)

