function model = construct_mseg_models(data_set,res)
%Construct a set of 'scene' models from a dataset of images and a
%covariance matrix

inds = round(linspace(1,length(data_set),100));
inds = unique(inds);

x = cell(length(inds),1);
for iii = 1:length(inds)
  i = inds(iii);
  
  I = toI(data_set{i});
  
  I2 = imresize_max(I,200);

  as = round(linspace(50,size(I2,1)-50,5));
  bs = round(linspace(50,size(I2,2)-50,5));
  
  % to show stuff
  params.display = 0;
  
  %fix these params to make a unary potential
  params.num_fg = 1;
  params.num_bg = 1;
  params.lambda = -1;
  params.beta = 30;
  params.fg_assignment = -50;
  
  % can scan over these vars the quickest since
  % they don't involve recomputation of unary
  [params.pb]= pbGM(I2, 2);

  params.Irow = reshape(RGB2Lab(I2),[],3);  
  [xxx,yyy] = meshgrid(1:size(I2,1),1:size(I2,2));
  xxx = (xxx - mean(xxx(:)))/std(xxx(:));
  yyy = (yyy - mean(yyy(:)))/std(yyy(:));
  params.Irow=cat(2,params.Irow,cat(2,reshape(xxx,[],1),reshape(yyy,[],1)));


  %the occupied region is always BG and nothing is learned there
  params.occupied = logical(zeros(size(params.pb)));
  ranger = -40:40;
  masks = cell(0,1);
  
  for a = 1:length(as)
    for b = 1:length(bs)
      fprintf(1,'.');
      mask = I2(:,:,1)*0;
      mask(as(a)+ranger,bs(b)+ranger)=1;
      
      r = mask;
      for asdf = 1:5
        r = gc3(I2,r,params);
      end
      

      %figure(34)
      %imagesc(segImage(segImage(I2,mask),r,[0 1 0]))
      %drawnow
      masks{end+1} = r;
      
    end
  end
    
 
  
  
  clear s objects
  s.I = I;

  for t = 1:length(masks)
    objects(t).class = sprintf('proposal',iii);
    objects(t).truncated = 0;
    objects(t).difficult = 0;
    [u,v] = find(imresize(masks{t},[size(I,1) size(I,2)],'nearest'));
    objects(t).bbox = [min(v) min(u) max(v) max(u)];
  end
  s.objects = objects;
  data_set2{iii} = s;
end

model = learnAllEG(data_set2,'',res);

model.params.detect_add_flip = 0;
model.params.detect_max_windows_per_exemplar = 1;
model.params.max_image_size = 200;
