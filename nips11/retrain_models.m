function retrain_models(models)
%Retrain models by loading shared negatives

VOCinit;
if ~exist('models','var')
  models = load_all_models('train','exemplars-svm');
end

nsv = cellfun2(@(x)x.model.nsv,models);
nsv = cat(2,nsv{:});

svids = cellfun2(@(x)x.model.svids,models);
svids = cat(2,svids{:});

psv = cellfun2(@(x)x.model.target_x,models);
psv = cat(2,psv{:});

pvids = cellfun2(@(x)x.model.target_id',models);
pvids = cat(2,pvids{:});

nsv = cat(2,nsv,psv);
svids = cat(2,svids,pvids);

mining_params = get_default_mining_params;
%% memex needs validation nodes
mining_params.extract_negatives = 1;
mining_params.SVMC = .01;
mining_params.SVMC = .01; %.01;

mining_params.max_negatives = 1000;

mining_params.BALANCE_POSITIVES = 0;
bg = get_pascal_bg('trainval');

%mining_params.DOMINANT_GRADIENT_PROJECTION = 1;
%mining_params.DOMINANT_GRADIENT_PROJECTION_K = 5;

myRandomize;
rrr = randperm(length(models));

final_directory = ...
    sprintf('%s/exemplars-memex/',VOCopts.localdir);

if ~exist(final_directory,'dir')
  mkdir(final_directory);
end

for indexi = 1:length(models)
  index = rrr(indexi);
  
  %index = 45;
  % index = 163;
  % index = 16;
  % index = 61;
  % index = 128;
  % index = 201;
  % index = 19;
  
  m = models{index};
  filer = sprintf('%s/%s.%s.%d.mat',final_directory,...
                  m.cls,m.curid,m.objectid);
  filerlock = [filer '.lock']
  if fileexists(filer) || (mymkdir_dist(filerlock) == 0)
    continue
  end
  
  NITER = 3;
  xstart = m.model.x(:,1);
  for bbb = 1:NITER

    m = add_new_detections(m,nsv,svids);
    m = cap_to_K_dets(m,2000);
    
    m.model.xstart = xstart;
    %m = do_svm(m,mining_params);
    m = do_rank(m,mining_params);
    [negatives,vals,pos,m] = find_set_membership(m);
    
    % curfeats = reshape(m.model.x,m.model.hg_size);
    % m.model.w = m.model.x*0;
    % mask3 = m.model.mask;
    % mask3 = mask3(:);
    % m.model.w(mask3) = curfeats(mask3) - mean(curfeats(mask3));
    % m.model.w = reshape(m.model.w,m.model.hg_size);
    
    
    Isv1 = get_sv_stack(m,bg,12,12);
    
    imwrite(Isv1,sprintf('%s/%s.%d_iter_I=%05d.png', ...
                         final_directory,m.curid,...
                         m.objectid,bbb),'png');
    figure(1)
    clf
    imagesc(Isv1)
    figure(2)
    clf
    show_cool_os(m)
    
    
    drawnow
    
    
    
    
    if bbb == NITER
      break;
    end
    %figure(1)
    %clf
    %imagesc(Isv1)
    %title(sprintf('%s.%d',m.curid,m.objectid))
    %drawnow
    
    %% now assimilate all positives to maximize gain vector
    keepers = nms_objid(m.model.svids);
    m.model.svids = m.model.svids(keepers);
    m.model.nsv = m.model.nsv(:,keepers);
    
    [negatives,vals,pos] = find_set_membership(m);
    [overlaps,matchind,matchclass] = get_overlaps_with_gt(m, ...
                                                  m.model.svids, ...
                                                  bg);
    
    VOCinit;
    targetc = find(ismember(VOCopts.classes,m.cls));
    gainvec = overlaps + .5*(matchclass == targetc);
    gainvec(overlaps < .5) = -5;
    gainvec(gainvec>.8) = gainvec(gainvec>.8) + .2;
    
    [score,ind] = max(cumsum(gainvec));
    fprintf(1,'ind for memex growth is %d\n',ind);
    set = 1:ind;
    set = set(overlaps(set)>.5);
    
    %[aaa,bbb] = sort(gainvec(pos),'descend');
    %set = pos(bbb(1:min(length(bbb),100)));
    %set = pos(gainvec(pos)>1);
    
    m.model.x = m.model.nsv(:,set);
    m.model.xinfo = m.model.svids(set);
    models{index} = m;
    
    maxpos = max(m.model.w(:)'*m.model.x - m.model.b);
    fprintf(1,' --- After assimilation positive is %.3f\n',maxpos);
    
    
    
    %figure(2)
    %clf
    %show_cool_os(m)
    
  end
  save(filer,'m');
  rmdir(filerlock);
end