function exemplar_initialize(e_set, init_function, init_params, ...
                             dataset_params, models_name)
% Initialize script which writes out initial model files for all
% exemplars in an exemplar stream e_set (see get_pascal_stream)
% is parallelizable (and dalalizable!)  
%
% There are several different initialization modes
% DalalMode:  Warp instances to mean frame from in-class exemplars
% Globalmode: Warp instances to hardcoded [8 8] frame
% new10mode:  Use a canonical 8x8 framing of each exemplar (this
%             allows for negative sharing in the future)
%
% cls: VOC class to process
% mode: mode name 
%
% Tomasz Malisiewicz (tomasz@cmu.edu)

%Only allow display to be enabled on a machine with X
[v,r] = unix('hostname');
if strfind(r, dataset_params.display_machine)==1
  display = 1;
else
  display = 0;
end

% DTstring = '';
% if dalalmode == 1
%   %Find the best window size from taking statistics over all
%   %training instances of matching class
%   hg_size = get_hg_size(cls);
%   DTstring = '-dt';
% elseif dalalmode == 2
%   hg_size = [8 8];
%   DTstring = '-gt';
% end
  
%if (dalalmode == 1) || (dalalmode == 2)
%  %Do the dalal-triggs anisotropic warping initialization
%  model = initialize_model_dt(I,bbox,SBIN,hg_size);
%else

results_directory = ...
    sprintf('%s/models/%s/',dataset_params.localdir, models_name);

if ~exist(results_directory,'dir')
  fprintf(1,'Making directory %s\n',results_directory);
  mkdir(results_directory);
end

fprintf(1,'Writing %d Exemplars to directory %s\n',length(e_set),...
        results_directory);

%Randomize creation order
myRandomize;
rrr = randperm(length(e_set));
e_set = e_set(rrr);

for i = 1:length(e_set)
 
  I = convert_to_I(e_set{i}.I);
  cls = e_set{i}.cls;
  objectid = e_set{i}.objectid;    
  anno = e_set{i}.anno;
  bbox = e_set{i}.bbox;

  [tmp,curid,tmp] = fileparts(e_set{i}.I);
  
  fprintf(1,'.');
    
  filer = sprintf('%s/%s.%d.%s.mat',...
                  results_directory, curid, objectid, cls);
  filerlock = [filer '.lock'];
  
  if fileexists(filer) || (mymkdir_dist(filerlock)==0)
    continue
  end
  gt_box = bbox;


  %Call the init function which is a mapping from (I,bbox) to (model)
  [model] = init_function(I,bbox,init_params);
  clear m
  m.model = model;    

  %Save filename (or original image)
  m.I = e_set{i}.I;
  m.curid = curid;
  m.objectid = objectid;
  m.cls = cls;    
  m.gt_box = gt_box;
  
  m.sizeI = size(I);
  m.models_name = models_name;
  
  save(filer,'m');
  if exist(filerlock,'dir')
    rmdir(filerlock);
  end

  %Print the bounding box overlap between the initial window and
  %the final window
  finalos = getosmatrix_bb(m.gt_box, m.model.bb(1,:));
  fprintf(1,'Final OS is %.3f\n', ...
          finalos);
  
  fprintf(1,'Final hg_size is %d %d\n',...
          m.model.hg_size(1), m.model.hg_size(2));
  
  
  if display == 1
    figure(1)
    clf
    subplot(1,3,1)
    imagesc(I)
    plot_bbox(m.model.bb(1,:),'',[1 0 0],[0 1 0],0,[1 3],m.model.hg_size)
    plot_bbox(m.gt_box,'',[0 0 1])
    axis image
    axis off
    title(sprintf('Exemplar %s.%d %s',m.curid,m.objectid,cls))
    
    subplot(1,3,2)
    imagesc(m.model.mask);
    axis image
    axis off
    grid on
    title(sprintf('%d x %d Mask',m.model.hg_size(1),m.model.hg_size(2)))
    
    subplot(1,3,3)      
    imagesc(HOGpicture(repmat(m.model.mask,[1 1 features]).*m.model.w))
    axis image
    axis off
    grid on
    title('Exemplar HOG features')
    drawnow
    
  end
end  
