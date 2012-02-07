function results = evaluateModel(data_set, test_struct, model)
params = model.params;

results = esvm_evaluate_pascal_voc(test_struct, data_set, model, params);

