function supertrain_models(models)
%Retrain ESVM models via VisRegression

VOCinit;
if ~exist('models','var')
  models = load_all_models;
end

%Load all those models
%am = load_all_models(models{1}.cls,'e100');

mining_params = get_default_mining_params;
mining_params.extract_negatives = 1;
myRandomize;
rrr = randperm(length(models));

final_directory = ...
    sprintf('%s/%s-vregmine/',...
            VOCopts.localdir,...
            models{1}.models_name);


if ~exist(final_directory,'dir')
  mkdir(final_directory);
end

BASEDIR = sprintf('%s/grids/',VOCopts.localdir);

for indexi = 1:length(models)
  index = rrr(indexi);

  %index = 5;
  %index = 16;
 
  %index = 68; 
  %index = 38;
  m = models{index};

  filer = sprintf('%s/%s.%s.%d.mat',final_directory,...
                  m.cls,m.curid,m.objectid);
  filerlock = [filer '.lock']
  if fileexists(filer) || (mymkdir_dist(filerlock) == 0)
    continue
  end
  %fprintf(1,'warn lock disabled\n');
  
  gridfiler = sprintf('%s/%s-%s.%05d.mat',...
                      BASEDIR,...
                      models{1}.cls,...
                      models{1}.models_name,...
                      index);
  
  grid = load(gridfiler);
  cb = grid.coarse_boxes;

  %   disp('hack load')  
  if 0
    load mym.mat
  else

    
    [ids1,nsv1] = extract_svs(cb,100,'train',sprintf('-%s',m.cls));
    [ids2,nsv2] = extract_svs(cb,100,'val',sprintf('-%s',m.cls));
    [ids3,nsv3] = extract_svs(cb,100,'trainval',sprintf('%s', ...
                                                  m.cls));
    m.model.svids = {};
    m.model.nsv = [];
    
    m.model.svids = cat(2,ids1,ids2,ids3,m.model.svids);%,hn.objids{1});
    m.model.nsv = cat(2,nsv1,nsv2,nsv3,m.model.nsv);%,hn.xs{1});
    
    [a,b,c,d,indicator] = find_set_membership(m.model.svids,m.cls);
    for i = 1:length(m.model.svids)
      m.model.svids{i}.set = indicator(i);
    end
    
    fprintf(1,'getting overlaps with gt\n');
    tic
    [Amaxos,Amaxind,Amaxclass] = ...
        get_overlaps_with_gt(m);
    
    for q = 1:length(m.model.svids)
      m.model.svids{q}.maxos = Amaxos(q);
      m.model.svids{q}.maxind = Amaxind(q);
      m.model.svids{q}.maxclass = Amaxclass(q);
    end
    toc
    
    %save mym.mat m
  end

  %get initial hog
  %m.model.w = reshape(m.model.x(:,1) - mean(m.model.x(:,1)),...
  %                    m.model.hg_size);

  %[aa,bb] = sort(cb(:,end),'descend');
  %nsv = cellfun2(@(x)x.model.nsv,models);
  %nsv = cat(2,nsv{:});
  %svids = cellfun2(@(x)x.model.svids,models);
  %svids = cat(2,svids{:});
  
  NITER = 10;
  %xstart = m.model.x(:,1);
  
  %savem = m;
  
  keepers = nms_objid(m.model.svids,.2);
  fprintf(1,'after nms keepers is %d\n',length(keepers));
  m.model.svids = m.model.svids(keepers);
  m.model.nsv = m.model.nsv(:,keepers);
  %print iteration 0
  % Isv1 = get_sv_stack(m,12);    
  % imwrite(Isv1,sprintf('%s/%s.%d_iter_I=%05d.png', ...
  %                      final_directory,m.curid,...
  %                      m.objectid,0),'png');
  
  fgall = cat(1, get_pascal_bg('train',['-' m.cls]),...
              get_pascal_bg('trainval',m.cls));
   
  %mining_queue = initialize_mining_queue(fg);
  for bbb = 1:NITER    
    %m.model.svids = savem.model.svids;
    %m.model.nsv = savem.model.nsv;
    [aa,bb] = sort(m.model.w(:)'*m.model.nsv,'descend');

    if 1 
      %START ENHANCE      
      curids = cellfun(@(x)x.curid,m.model.svids(bb(1:50)));
      clear fg
      fg = cell(length(curids),1);
      for z = 1:length(curids)
        fg{z} = sprintf(VOCopts.imgpath,sprintf('%06d',curids(z)));
      end
      rrr2 = randperm(length(fgall));
      fg = cat(1,fg,fgall(rrr2(1:50)));
      fg = unique(fg);
      mining_queue = initialize_mining_queue(unique(fg));
      %mining_queue = mining_queue(1:10);
      
      mining_params = get_default_mining_params;
      mining_params.SAVE_SVS = 1;
      mining_params.detection_threshold = -1.0;
      mining_params.thresh = -1.0;
      
      mining_params.TOPK = 5;
      mining_params.MAX_WINDOWS_BEFORE_SVM = 200;
      mining_params.MAX_IMAGES_BEFORE_SVM = 100;
      mining_params.NMS_MINES_OS = 1.0;
      
      curm = m;
      
      %THIS HACK must be done on a copy (it is a hack to get more dets)
      curm.model.b = curm.model.b - 0.1;
      [hn, mining_queue, mining_stats] = ...
          load_hn_fg({curm}, mining_queue, fg, mining_params);
      
      %%% END ENHANCE
        
      fprintf(1,'getting overlaps with gt\n');
      tic
      [Amaxos,Amaxind,Amaxclass] = ...
          get_overlaps_with_gt(m,hn.objids{1});
      
      for q = 1:length(hn.objids{1})
        hn.objids{1}{q}.maxos = Amaxos(q);
        hn.objids{1}{q}.maxind = Amaxind(q);
        hn.objids{1}{q}.maxclass = Amaxclass(q);
      end
      toc
      
      %Concatenate detections with model stuff
      m.model.svids = cat(2,m.model.svids, hn.objids{1});
      m.model.nsv   = cat(2,m.model.nsv, hn.xs{1});
      
      [a,b,c,d,indicator] = find_set_membership(m.model.svids,m.cls);
      for i = 1:length(m.model.svids)
        m.model.svids{i}.set = indicator(i);
      end     
    end
    
    [aa,bb] = sort(m.model.w(:)'*m.model.nsv,'descend');
        
    m.model.svids = m.model.svids(bb);
    m.model.nsv = m.model.nsv(:,bb);
    
    keepers = nms_objid(m.model.svids,.2);
    fprintf(1,'after nms keepers is %d\n',length(keepers));
    m.model.svids = m.model.svids(keepers);
    m.model.nsv = m.model.nsv(:,keepers);
    m = cap_to_K_dets(m,2000);
    
    
    mold = m;
    m = do_rank(m,mining_params);
    
    % mold.model.svids = m.model.svids;
    % mold.model.nsv = m.model.nsv;
    
    % Isv1 = get_sv_stack(mold,12);
    % imwrite(Isv1,sprintf('%s/%s.%d_iter_I=%05d-oldw.png', ...
    %                      final_directory,m.curid,...
    %                      m.objectid,bbb),'png');
    
    figure(1)
    show_cool_os(m)
    drawnow
    
    if (mining_params.dump_images == 1) || ...
          (mining_params.dump_last_image == 1 && ...
           m.iteration == mining_params.MAXITER)
      set(gcf,'PaperPosition',[0 0 20 5]);
      print(gcf,sprintf('%s/%s.%d_iter=%05d.png', ...
                        final_directory,m.curid,...
                        m.objectid,bbb),'-dpng'); 
      
      Isv1 = get_sv_stack(m,12);
      imwrite(Isv1,sprintf('%s/%s.%d_iter_I=%05d.png', ...
                           final_directory,m.curid,...
                           m.objectid,bbb),'png');
    end
           
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
  
  %m.models_name = [m.models_name '-rereg'];
  save(filer,'m');
  if exists(filerlock,'dir')
    rmdir(filerlock);
  end
end


