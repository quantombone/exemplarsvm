%Here is a quick demo

addpath(genpath(pwd))

[models,M,test_set] = esvm_download_models('bus');
esvm_demo_apply_exemplars(test_set(1:10), models, M);

[models,M,test_set] = esvm_download_models('bicycle');
esvm_demo_apply_exemplars(test_set(1:10), models, M);

[models,M,test_set] = esvm_download_models('train');
esvm_demo_apply_exemplars(test_set(1:10), models, M);
