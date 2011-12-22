function bg = get_james_bg(LEN,subset)
%Create a virtual 'bg' dataset from James Hays' 6.5 million dataset
%in canonical dataset order (which is pretty random)

if ~exist('LEN','var')
  LEN = 1000;
end

if ~exist('subset','var')
  bg = cell(LEN,1);
  for i = 1:LEN
    bg{i} = sprintf('load_james_image(%d)',i);
  end
else
  LEN = min(LEN,length(subset));
  bg = cell(LEN,1);
  for i = 1:LEN
    bg{i} = sprintf('load_james_image(%d)',subset(i));
  end
end
