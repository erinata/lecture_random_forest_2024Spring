import pandas
import kfold_template

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

dataset = pandas.read_csv("temperature_data.csv")
dataset = pandas.get_dummies(dataset)
dataset = dataset.sample(frac=1).reset_index()

target = dataset["actual"].values
data = dataset.drop(["actual","level_0","temp_lag1"], axis = 1)

feature_list = data.columns
data = data.values

print(feature_list)
print(target)
print(data)

machine = RandomForestClassifier(criterion="gini", max_depth=2, n_estimators=100, bootstrap = True) 
return_values = kfold_template.run_kfold(machine, data, target, 4, True)
print(return_values)


machine = RandomForestClassifier(criterion="gini", max_depth=2, n_estimators=100, bootstrap = True) 
machine.fit(data, target)
feature_importances_raw = machine.feature_importances_
print(feature_importances_raw)
print(feature_list)

feature_zip = zip(feature_list, feature_importances_raw)
print(feature_zip)

feature_importances = [ (feature, round(importance, 4)) for feature, importance in feature_zip]
feature_importances = sorted(feature_importances, key = lambda x: x[1] )
print(feature_importances)
[ print('{:14}: {}'.format(*feature_importance)) for feature_importance in feature_importances]
























