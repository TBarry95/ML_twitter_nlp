









'''
print("# -- Random Forest: Contribution for Decision (egs - All variables) -- #")
predictions_egs = data_test_lin[0:2]
prediction, bias, contributions = ti.predict(rf_model, predictions_egs)
for i in range(len(predictions_egs)):
    print("Prediction", i)
    print( "Contribution by Top Feature:")
    for c, feature in sorted(zip(contributions[i], data_test_lin.columns))[0:2]:
        print(feature, round(c, 2))
    print( "-"*20)
print("##########################################################")
print("##########################################################")'''

# -- 2. Predict Direction: Better than logistic?

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc.fit(data_reduced_train_log, price_train_log)
p = rfc.predict(data_reduced_test_log)
acc_rfc = rfc.score(data_reduced_test_log, price_test_log)
print("# -- Test Results - Random Forest Classifier: All Variables  -- #")
print("Mean accuracy: ", acc_rf_pca_dir)
print("##########################################################")
print("##########################################################")




