% function model = populate_wiggles(I, model, NWIGGLES)
% %Get wiggles
% xxx = replica_hits(I, model.params.sbin, model.target_id, ...
%                    model.hg_size, NWIGGLES);

% %% GET self feature + NWIGGLES wiggles "perturbed images"
% model.x = xxx;
% model.w = reshape(mean(model.x,2), model.hg_size);
% model.w = model.w - mean(model.w(:));
% model.b = -100;
