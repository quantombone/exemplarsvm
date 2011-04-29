function Isv = get_sv_stack(svids, bg, m, K1, K2)
%Create an image which visualizes the negative support vectors
%Tomasz Malisiewicz (tomasz@cmu.edu)

if iscell(svids)
  svids = [svids{:}];
end

if exist('m','var')
  hogpic = (HOGpicture(m.model.w));
  hogpic = jettify(hogpic);
end


N = length(svids);

svids = svids(1:N);
ucurids = unique([svids.curid]);
svims = cell(N,1);

PADDER = 100;

for i = 1:length(ucurids)
  I = convert_to_I(bg{ucurids(i)});
  

  I = pad_image(I, PADDER);
  hits = find([svids.curid]==ucurids(i));
  for j = 1:length(hits)
    
    bb = round(svids(hits(j)).bb + PADDER);
    try
      svims{hits(j)} = I(bb(2):bb(4),bb(1):bb(3),:);
    catch
      
      svims{hits(j)} = rand(bb(4)-bb(2)+1,bb(3)-bb(1)+1,3);
    end
   
    if svids(hits(j)).flip == 1
      svims{hits(j)} = flip_image(svims{hits(j)});
    end
  end
end

%resize to average size
% newsize = ([mean(cellfun(@(x)size(x,1),svims)) mean(cellfun(@ ...
%                                                   (x)size(x,2), ...
%                                                   svims))]);
% newsize = 100/newsize(1) * newsize;
% newsize = round(newsize);
VOCinit;
Ibase = imread(sprintf(VOCopts.imgpath,m.curid));
Ibase = im2double(Ibase);
Ibase = Ibase(m.gt_box(2):m.gt_box(4),m.gt_box(1):m.gt_box(3),:);
newsize = [size(Ibase,1) size(Ibase,2)];

newsize = 100/newsize(1) * newsize;
newsize = round(newsize);

svims = cellfun2(@(x)max(0.0,min(1.0,imresize(x,newsize))),svims);

%K = ceil(sqrt(N));
%K2 = 1;
%K1 = 12;

%K1 = K;
%K2 = K;
if length(svims)<K1*K2
  svims{K1*K2} = zeros(newsize(1),newsize(2),3);
end

for j = (N+1):(K1*K2)
  svims{j} = zeros(newsize(1),newsize(2),3);
end
sstack = cat(4,svims{:});

mx = mean(m.model.x,2);
raw_picture = HOGpicture(reshape(mx-mean(mx(:)),m.model.hg_size));
pos_picture = HOGpicture(m.model.w);
neg_picture = HOGpicture(-m.model.w);
spatial_picture = sum(m.model.w.*reshape(mean(m.model.x,2), ...
                                         size(m.model.w)),3);
spatial_picture = imresize(spatial_picture,[size(pos_picture,1) ...
                    size(pos_picture,2)],'nearest');

raw_picture = jettify(imresize(raw_picture,newsize,'nearest'));
pos_picture = jettify(imresize(pos_picture,newsize,'nearest'));
neg_picture = jettify(imresize(neg_picture,newsize,'nearest'));
spatial_picture = jettify(imresize(spatial_picture,newsize,'nearest'));


%% compute the mean image
ms = mean(sstack,4);

svims(6:end) = svims(1:end-5);

%ex goes in slot 1
svims{1} = max(0.0,min(1.0,imresize(Ibase,[size(ms,1),size(ms, ...
                                                  2)])));
svims{2} = pos_picture;
svims{3} = neg_picture;
svims{4} = spatial_picture;
svims{5} = ms;
%svims{3} = max(0.0,min(1.0,imresize(hogpic,[size(ms,1),size(ms, ...
%                                                  2)])));
%VOCinit;

svims = reshape(svims,K1,K2)';
for j = 1:K2
  svrows{j} = cat(2,svims{j,:});
end

Isv = cat(1,svrows{:});

