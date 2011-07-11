function model = populate_wiggles(I, model, NWIGGLES)
%Get wiggles
xxx = replica_hits(I, model.init_params.sbin, model.bb(1,:), ...
                   model.hg_size, NWIGGLES, model);

%% GET self feature + NWIGGLES wiggles "perturbed images"
model.x = xxx;
model.w = reshape(mean(model.x,2), model.hg_size);
model.w = model.w - mean(model.w(:));
%model.b = -100;
model.b = 0;
