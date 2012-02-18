function results = evaluateModel(data_set, boxes, cls)


results = esvm_evaluate_pascal_voc(boxes, data_set, model, params);

