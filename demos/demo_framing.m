%% Showcase of Exemplar Framings
% In this example, we initialize some exemplars with different
% framing strategies.

% To generate this pdf, run publish('demo_framing','pdf');
cls = 'cow';
VOCYEAR = 'VOC2007';

workdir = '/nfs/baikal/tmalisie/inits11/';

%Initialize dataset
dataset_params = get_voc_dataset(VOCYEAR, workdir);

%Initialize exemplar stream with 3 exemplars
stream_set_name = 'trainval';
stream_max_ex = 3;
e_stream_set = get_pascal_exemplar_stream(dataset_params, ...
                                          stream_set_name, ...
                                          cls, stream_max_ex);

%Initialize scene stream with 3 scenes
e_scene_stream_set = get_pascal_scene_stream(dataset_params, ...
                                             stream_set_name, ...
                                             cls, stream_max_ex);

%% Scene: GoalSize goal_ncells = [100], MAXDIM=[10]
% This mode will try create a framing which is made up of a target
% number of cells, subject to one dimension being at most MAXDIM
clear init_params;
init_params.sbin = 8;
init_params.goal_ncells = 100;
init_params.MAXDIM = 10;
init_params.init_type = 'goalsize';
init_params.init_function = @initialize_goalsize_model;
models_name = ['demo_scene_goalsize'];
%Initialize exemplars with the exemplar stream
efiles = exemplar_initialize(dataset_params, e_scene_stream_set, ...
                             models_name, init_params);

models = load_all_models(dataset_params, cls, models_name, ...
                         efiles);

show_exemplar_frames(models, 5, dataset_params);
snapnow;


%% Scene: GoalSize goal_ncells = [300], MAXDIM=[15]
% Here is the above example, but allowing for a much finer scene
% representation

clear init_params;
init_params.sbin = 8;
init_params.goal_ncells = 300;
init_params.MAXDIM = 15;
init_params.init_type = 'goalsize';
init_params.init_function = @initialize_goalsize_model;
models_name = ['demo_scene_goalsize_maxdim'];

%Initialize exemplars with the exemplar stream
efiles = exemplar_initialize(dataset_params, e_scene_stream_set, ...
                             models_name, init_params);

models = load_all_models(dataset_params, cls, models_name, ...
                         efiles);

show_exemplar_frames(models, 5, dataset_params);
snapnow;

%Delete files (only for demo)
for i = 1:length(efiles)
 delete(efiles{i});
end

%% Exemplars: GoalSize goal_ncells = [100], MAXDIM=[10]
% Experiment repeated for exemplars
clear init_params;
init_params.sbin = 8;
init_params.goal_ncells = 100;
init_params.MAXDIM = 10;
init_params.init_type = 'goalsize';
init_params.init_function = @initialize_goalsize_model;
models_name = ['demo_ex_goalsize'];

%Initialize exemplars with the exemplar stream
efiles = exemplar_initialize(dataset_params, e_stream_set, ...
                             models_name, init_params);

models = load_all_models(dataset_params, cls, models_name, ...
                         efiles);

show_exemplar_frames(models, 5, dataset_params);
snapnow;

%Delete files (only for demo)
for i = 1:length(efiles)
 delete(efiles{i});
end

%% Exemplar: FixedFrame hg_size = [8 8]
% Experiment for fixed-size exemplars
clear init_params;
init_params.sbin = 8;  
init_params.hg_size = [8 8];
init_params.init_type = 'fixedframe';
init_params.init_function = @initialize_fixedframe_model;
models_name = ['demo_ex_ff'];

%Initialize exemplars with the exemplar stream
efiles = exemplar_initialize(dataset_params, e_stream_set, ...
                             models_name, init_params);

models = load_all_models(dataset_params, cls, models_name, ...
                         efiles);

show_exemplar_frames(models, 5, dataset_params);
snapnow;

%Delete files (only for demo)
for i = 1:length(efiles)
  delete(efiles{i});
end

