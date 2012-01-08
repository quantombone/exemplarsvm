
esvm_download_models('bus');
load voc2007-bus.mat
esvm_demo_apply_exemplars(test_set(1:10), models, M);

esvm_download_models('bicycle');
load voc2007-bicycle.mat
esvm_demo_apply_exemplars(test_set(1:10), models, M);

esvm_download_models('train');
load voc2007-train.mat
esvm_demo_apply_exemplars(test_set(1:10), models, M);
