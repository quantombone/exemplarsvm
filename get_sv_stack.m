function Isv = get_sv_stack(m, bg, K1, K2)
%% Create a K1xK2 image which visualizes the negative support vectors as
%% well as the exemplar learned HOG, and averaged SVs
%% Tomasz Malisiewicz (tomasz@cmu.edu)

if ~exist('K2','var')
  K2 = K1;
end

%% sort by score
r = m.model.w(:)'*m.model.nsv - m.model.b;
[aa,bb] = sort(r,'descend');
m.model.svids = m.model.svids(bb);
m.model.nsv = m.model.nsv(:,bb);

svids = m.model.svids;

if exist('m','var')
  hogpic = (HOGpicture(m.model.w));
  hogpic = jettify(hogpic);
end

N = length(svids);
N = min(N,K1*K2);

svids = svids(1:N);
svids = [svids{:}];

if N > 0
  ucurids = unique([svids.curid]);
else
  ucurids = [];
end
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

VOCinit;
try
  Ibase = imread(sprintf(VOCopts.imgpath,m.curid));
  Ibase = im2double(Ibase);
  PADDER = 100;
  Ibase = pad_image(Ibase,PADDER);
  cb = m.model.coarse_box+PADDER;
  Ibase = Ibase(round(cb(2):cb(4)), round(cb(1):cb(3)),:);
catch
  Ibase = zeros(m.model.hg_size(1)*10,m.model.hg_size(2)*10,3);
end

newsize = [size(Ibase,1) size(Ibase,2)];
newsize = 100/newsize(1) * newsize;
newsize = round(newsize);

svims = cellfun2(@(x)max(0.0,min(1.0,imresize(x,newsize))),svims);

[negatives,vals,pos] = find_set_membership(m);

svimstack = cat(4,svims{:});
NMS = K2-4;
cuts = round(linspace(1,size(svimstack,4),NMS));
for i = 1:length(cuts)
  mss{i} = mean(svimstack(:,:,:,1:cuts(i)),4);
end
for i = 1:length(svims)
  %% find membership here
  if sum(i == negatives)
    svims{i} = pad_image(svims{i},-5);
    svims{i} = pad_image(svims{i},5,[0 0 0]);
  elseif sum(i == pos)
    svims{i} = pad_image(svims{i},-5);
    svims{i} = pad_image(svims{i},5,[1 0 0]);
  else
    svims{i} = pad_image(svims{i},-5);
    svims{i} = pad_image(svims{i},5,[0 0 1]);
  end
end

if length(svims)<K1*K2
  svims{K1*K2} = zeros(newsize(1),newsize(2),3);
end

for j = (N+1):(K1*K2)
  svims{j} = zeros(newsize(1),newsize(2),3);
end

%sstack = cat(4,svims{:});
%% compute the mean image
%ms = mean(sstack,4);

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


NSHIFT = length(mss) + 4;


svims((NSHIFT+1):end) = svims(1:end-NSHIFT);

%ex goes in slot 1
svims{1} = max(0.0,min(1.0,imresize(Ibase,[size(pos_picture,1),size(pos_picture, ...
                                                  2)])));
svims{2} = pos_picture;
svims{3} = neg_picture;
svims{4} = spatial_picture;
svims(4+(1:length(mss))) = mss;
%svims{3} = max(0.0,min(1.0,imresize(hogpic,[size(ms,1),size(ms, ...
%                                                  2)])));
%VOCinit;

%looks bad with padding
%svims = cellfun2(@(x)pad_image(x,2,[1 1 1]),svims);

svims = reshape(svims,K1,K2)';
for j = 1:K2
  svrows{j} = cat(2,svims{j,:});
end

Isv = cat(1,svrows{:});
