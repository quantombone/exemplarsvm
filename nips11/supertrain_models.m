function supertrain_models(models)
%Retrain models by loading shared negatives

VOCinit;
if ~exist('models','var')
  models = load_all_models;
end

bg = get_pascal_bg('trainval');

mining_params = get_default_mining_params;
mining_params.extract_negatives = 1;
myRandomize;
rrr = randperm(length(models));

final_directory = ...
    sprintf('%s/exemplars-rereg/',VOCopts.localdir);

if ~exist(final_directory,'dir')
  mkdir(final_directory);
end

for indexi = 1:length(models)
  index = rrr(indexi);
  
  %index = 16;
  
  m = models{index};

  filer = sprintf('%s/%s.%s.%d.mat',final_directory,...
                  m.cls,m.curid,m.objectid);
  filerlock = [filer '.lock']
  if fileexists(filer) || (mymkdir_dist(filerlock) == 0)
    continue
  end
  
  grid = load(sprintf('/nfs/baikal/tmalisie/grids/%s.%05d.mat', ...
                      models{1}.cls,index));
  cb = grid.coarse_boxes;

  m = try_reshape(m,cb,1000);

  %get initial hog
  %m.model.w = reshape(m.model.x(:,1) - mean(m.model.x(:,1)),...
  %                    m.model.hg_size);

  %[aa,bb] = sort(cb(:,end),'descend');
  %nsv = cellfun2(@(x)x.model.nsv,models);
  %nsv = cat(2,nsv{:});
  %svids = cellfun2(@(x)x.model.svids,models);
  %svids = cat(2,svids{:});
  
  NITER = 3;
  xstart = m.model.x(:,1);
  
  savem = m;
    
  for bbb = 1:NITER

    m.model.svids = savem.model.svids;
    m.model.nsv = savem.model.nsv;
    
    [aa,bb] = sort(m.model.w(:)'*m.model.nsv,'descend');
    m.model.svids = m.model.svids(bb);
    m.model.nsv = m.model.nsv(:,bb);
    keepers = nms_objid(m.model.svids,.2);
    fprintf(1,'after nms keepers is %d\n',length(keepers));
    m.model.svids = m.model.svids(keepers);
    m.model.nsv = m.model.nsv(:,keepers);
    %m = add_new_detections(m,nsv,svids);
    
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
    
    if (mining_params.dump_images == 1) || ...
          (mining_params.dump_last_image == 1 && ...
           m.iteration == mining_params.MAXITER)
      set(gcf,'PaperPosition',[0 0 20 5]);
      print(gcf,sprintf('%s/%s.%d_iter=%05d.png', ...
                        final_directory,m.curid,...
                        m.objectid,bbb),'-dpng'); 
    end

    
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
    
    % %% now assimilate all positives to maximize gain vector
    % keepers = nms_objid(m.model.svids);
    % m.model.svids = m.model.svids(keepers);
    % m.model.nsv = m.model.nsv(:,keepers);
    

    % [negatives,vals,pos] = find_set_membership(m);
    % [overlaps,matchind,matchclass] = get_overlaps_with_gt(m, ...
    %                                               m.model.svids, ...
    %                                               bg);
    
    % VOCinit;
    % targetc = find(ismember(VOCopts.classes,m.cls));
    % gainvec = overlaps + .5*(matchclass == targetc);
    % gainvec(overlaps < .5) = -5;
    % gainvec(gainvec>.8) = gainvec(gainvec>.8) + .2;
    
    % [score,ind] = max(cumsum(gainvec));
    % fprintf(1,'ind for memex growth is %d\n',ind);
    % set = 1:ind;
    % set = set(overlaps(set)>.5);
    
    % %[aaa,bbb] = sort(gainvec(pos),'descend');
    % %set = pos(bbb(1:min(length(bbb),100)));
    % %set = pos(gainvec(pos)>1);
    
    % m.model.x = m.model.nsv(:,set);
    % m.model.xinfo = m.model.svids(set);
    % models{index} = m;
    
    % maxpos = max(m.model.w(:)'*m.model.x - m.model.b);
    % fprintf(1,' --- After assimilation positive is %.3f\n',maxpos);
    
    
    
    %figure(2)
    %clf
    %show_cool_os(m)
    
  end
  save(filer,'m');
  rmdir(filerlock);
end