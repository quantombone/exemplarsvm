function I = create_grid_res(ids,NNN,sizer,superind,relatedinds)
%Create a grid of resulting images all resized to 100x100 size
%res is the resulting scores (sorted in descending order)
if ~exist('NNN','var')
  NNN = 5;
end

stacker1 = cell(NNN,NNN);

if ~exist('sizer','var')
  sizer = [100 100];
end
if ~exist('superind','var')
  superind = -1;
end
if ~exist('relatedinds','var')
  relatedinds = -1;
end
for j = 1:NNN*NNN

  try
    baseI = convert_to_I(ids{j});
    Icur = max(0.0,min(1.0,...
                       imresize(baseI,...
                                sizer)));
  catch
    Icur = zeros(sizer(1),sizer(2),3);
  end
  

  if j == superind
    Icur = pad_image(Icur,-5);
    Icur = pad_image(Icur,5,[1 0 0]);
  elseif sum(ismember(j,relatedinds))>0
    Icur = pad_image(Icur,-5);
    Icur = pad_image(Icur,5,[0 1 0]);
  end
  
  stacker1{j} = Icur;
end

clear sss;
for j = 1:size(stacker1,1)
  sss{j} = cat(2,stacker1{:,j}); 
end
sss = cat(1,sss{:});
I = sss;
